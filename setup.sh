#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
env_file="$repo_root/.env"

if [[ ! -f "$env_file" ]]; then
    echo "No .env at $env_file" >&2
    exit 1
fi

set -a
source "$env_file"
set +a

: "${DB_NAME:?DB_NAME missing from .env}"
: "${DB_USER:?DB_USER missing from .env}"
: "${DB_PASSWORD:?DB_PASSWORD missing from .env}"

sudo -u postgres psql \
    --no-psqlrc \
    --set ON_ERROR_STOP=1 \
    --set app_role="$DB_USER" \
    --set app_password="$DB_PASSWORD" \
    --set database_name="$DB_NAME" \
    < "$repo_root/db/bootstrap/001_create_roles_and_database.sql"

for migration in "$repo_root"/db/migrations/*.sql; do
    PGPASSWORD="$DB_PASSWORD" psql \
        --no-psqlrc \
        --set ON_ERROR_STOP=1 \
        --host "${DB_HOST:-localhost}" \
        --port "${DB_PORT:-5432}" \
        --username "$DB_USER" \
        --dbname "$DB_NAME" \
        --file "$migration"
done
