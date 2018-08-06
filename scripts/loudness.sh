#!/bin/bash

main()
{
    local file

    while [[ $# -gt 0 ]]
    do
        file="$1"
        print_summary "$file"
        shift
    done
}

print_summary()
{
    local file="$1"
    local summary
    local integrated
    local i_threshold
    local lra
    local lra_threshold
    local lra_low
    local lra_high
    local peak

    summary=$(
        ffmpeg \
            -nostats \
            -i "$file" \
            -filter_complex '[a:0]ebur128=peak=true' \
            -f null - 2>&1 \
            | tail -n 12)

    integrated=$(extract_from_summary "I" "$summary")
    i_threshold=$(extract_from_summary "Threshold" "$summary" | head -n1)
    lra=$(extract_from_summary "" "$summary")
    lra_threshold=$(extract_from_summary "Threshold" "$summary" | tail -n1)
    lra_low=$(extract_from_summary "LRA low" "$summary")
    lra_high=$(extract_from_summary "LRA high" "$summary")
    peak=$(extract_from_summary "Peak" "$summary")

    printf "I: %s\tT: %s\tLRA: %s\tL: %s\tT: %s\tH: %s\tPeak: %s\t%s\n" \
        "$integrated" "$i_threshold" \
        "$lra" "${lra_threshold}" "${lra_low}" "${lra_high}" \
        "$peak" "$file"

    return 0
}

extract_from_summary()
{
    local field="$1"
    local summary="$2"

    printf "%s" "$summary" \
        | grep "^ *$field: " \
        | cut -d":" -f2 \
        | sed "s/^ *//"
}

main "$@"
exit $?
