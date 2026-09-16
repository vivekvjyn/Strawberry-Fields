#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

if [ ! -f .env ]; then
    echo "Missing .env - copy the DB_HOST/DB_NAME/DB_USER/DB_PASSWORD/DB_PORT/SECRET_KEY vars from the README first." >&2
    exit 1
fi

set -a
source .env
set +a

psql_run() {
    if [ -n "${DATABASE_URL:-}" ]; then
        psql "$DATABASE_URL" -v ON_ERROR_STOP=1 -qtA "$@"
    else
        PGPASSWORD="$DB_PASSWORD" psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -v ON_ERROR_STOP=1 -qtA "$@"
    fi
}

sql_text() {
    local value="$1"
    if [ -z "$value" ]; then
        echo "NULL"
    else
        echo "'$(printf '%s' "$value" | sed "s/'/''/g")'"
    fi
}

echo "==> Creating schema"
psql_run -f "$SCRIPT_DIR/schema.sql"

DATA_HOME=""
DOWNLOAD=0
LIMIT=0

while [ $# -gt 0 ]; do
    case "$1" in
        --data-home)
            DATA_HOME="$2"
            shift 2
            ;;
        --download)
            DOWNLOAD=1
            shift
            ;;
        --limit)
            LIMIT="$2"
            shift 2
            ;;
        *)
            echo "Unknown argument: $1" >&2
            exit 1
            ;;
    esac
done

ZENODO_URL=$(grep -A4 '^saraga:' config.yaml | grep 'zenodo_url:' | awk '{print $2}' | tr -d '"')
CACHE_DIR=$(grep -A4 '^saraga:' config.yaml | grep 'cache_dir:' | awk '{print $2}')
HOP_SECONDS=$(grep -A4 '^pitch:' config.yaml | grep 'hop_seconds:' | awk '{print $2}')

if [ "$DOWNLOAD" -eq 1 ] || [ -z "$DATA_HOME" ]; then
    mkdir -p "$CACHE_DIR"
    zip_path="$CACHE_DIR/saraga1.5_carnatic.zip"
    tmp_path="$zip_path.tmp"

    if [ ! -f "$zip_path" ]; then
        echo "==> Downloading Saraga Carnatic 1.5 from Zenodo"
        curl -L --fail -C - -o "$tmp_path" "$ZENODO_URL"
        mv "$tmp_path" "$zip_path"
    fi

    marker="$CACHE_DIR/extracted"
    if [ ! -f "$marker" ]; then
        echo "==> Extracting archive"
        unzip -q "$zip_path" -d "$CACHE_DIR"
        touch "$marker"
    fi

    DATA_HOME="$CACHE_DIR"
fi

echo "==> Ingesting Saraga Carnatic from $DATA_HOME"

total=$(find "$DATA_HOME" -name '*.json' | wc -l)
echo "Found $total candidate tracks"

count=0
inserted=0

while IFS= read -r -d '' metadata_path; do
    track_dir=$(dirname "$metadata_path")
    name=$(basename "$metadata_path" .json)
    ctonic_path="$track_dir/$name.ctonic.txt"
    [ -f "$ctonic_path" ] || continue

    count=$((count + 1))
    if [ "$LIMIT" -gt 0 ] && [ "$count" -gt "$LIMIT" ]; then
        break
    fi

    pitch_file="$track_dir/$name.pitch-vocal.txt"
    [ -f "$pitch_file" ] || pitch_file="$track_dir/$name.pitch.txt"
    if [ ! -f "$pitch_file" ]; then
        echo "  [$count/$total] Skipping $name: no pitch annotation"
        continue
    fi

    tonic=$(awk -F'\t' '{print $1; exit}' "$ctonic_path")

    mapfile -t contour_output < <(awk -v tonic="$tonic" -v hop="$HOP_SECONDS" -f "$SCRIPT_DIR/pitch_contour.awk" "$pitch_file")
    if [ "${contour_output[0]}" = "EMPTY" ]; then
        echo "  [$count/$total] Skipping $name: no voiced pitch"
        continue
    fi
    duration="${contour_output[0]}"
    pitch_cents="${contour_output[1]}"

    title=$(jq -r '.title // empty' "$metadata_path")
    [ -n "$title" ] || title="$name"
    mbid=$(jq -r '.mbid // empty' "$metadata_path")
    artists=$(jq -r '[.artists[]?.artist | (.name // .title // .common_name)] | map(select(. != null)) | join(", ")' "$metadata_path")
    raaga=$(jq -r '[.raaga[]? | (.name // .title // .common_name)] | map(select(. != null)) | join(", ")' "$metadata_path")
    taala=$(jq -r '[.taala[]? | (.name // .title // .common_name)] | map(select(. != null)) | join(", ")' "$metadata_path")
    work=$(jq -r '[.work[]? | (.name // .title // .common_name)] | map(select(. != null)) | join(", ")' "$metadata_path")
    concert=$(jq -r '[.concert[]? | (.name // .title // .common_name)] | map(select(. != null)) | join(", ")' "$metadata_path")

    insert_sql="INSERT INTO tracks (mbid, title, artists, raaga, taala, work, concert, tonic, duration, hop_seconds, pitch_cents)
                VALUES ($(sql_text "$mbid"), $(sql_text "$title"), $(sql_text "$artists"), $(sql_text "$raaga"),
                        $(sql_text "$taala"), $(sql_text "$work"), $(sql_text "$concert"),
                        $tonic, $duration, $HOP_SECONDS, '$pitch_cents');"

    if error=$(psql_run -c "$insert_sql" 2>&1 >/dev/null); then
        inserted=$((inserted + 1))
        echo "  [$count/$total] Ingested $name"
    else
        echo "  [$count/$total] Skipping $name: $error"
    fi
done < <(find "$DATA_HOME" -name '*.json' -print0 | sort -z)

echo "==> Ingested $inserted/$count tracks into the database"
echo "==> Done. Run 'python wsgi.py' to start the server."
