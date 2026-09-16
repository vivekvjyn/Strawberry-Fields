BEGIN {
    FS = "\t"
    voiced_count = 0
    frame_count = 0
}

{
    frame_count++
    time = $1 + 0
    freq = $2 + 0
    times[frame_count] = time
    if (freq > 0) {
        voiced_count++
        voiced_times[voiced_count] = time
        voiced_cents[voiced_count] = 1200 * log(freq / tonic) / log(2)
    }
}

END {
    if (voiced_count == 0 || frame_count < 2) {
        print "EMPTY"
        exit
    }

    duration = times[frame_count]
    grid_points = int(duration / hop) + 1

    pointer = 1
    contour = "{"
    for (i = 0; i < grid_points; i++) {
        grid_time = i * hop

        while (pointer < voiced_count && voiced_times[pointer + 1] <= grid_time) {
            pointer++
        }

        if (grid_time <= voiced_times[1]) {
            cents = voiced_cents[1]
        } else if (grid_time >= voiced_times[voiced_count]) {
            cents = voiced_cents[voiced_count]
        } else {
            left_time = voiced_times[pointer]
            right_time = voiced_times[pointer + 1]
            left_cents = voiced_cents[pointer]
            right_cents = voiced_cents[pointer + 1]
            if (right_time == left_time) {
                cents = left_cents
            } else {
                fraction = (grid_time - left_time) / (right_time - left_time)
                cents = left_cents + fraction * (right_cents - left_cents)
            }
        }

        if (i > 0) {
            contour = contour ","
        }
        contour = contour sprintf("%.6f", cents)
    }
    contour = contour "}"

    print grid_points * hop
    print contour
}
