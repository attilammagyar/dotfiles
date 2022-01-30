#!/bin/bash

main()
{
    local file

    cat <<HELP
# LUFS = Loudness Unit Full Scale (Loudness K-weighted, relative to full scale)
# LU   = Relative Loudness Unit (1 LU ~ relative measurement of 1 dB)
#
# I    = Integrated loudness (average loudness over the whole program)
# IT   = Integrated loudness threshold
# LRA  = Loudness range (LH - LL)
# LT   = Loudness range threshold (20 LU below Absolute-Gated Loudness which is
#        the avg of short-term loudness values that exceed -70 LUFS, with 3s
#        sliding window)
# LL   = Loudness range low (quietest 10% above LT)
# LH   = Loudness range high (loudest 95% above LT)
# P    = True peak (inter-sample peaks)
#
HELP

    printf "I\tIT\tLRA\tLT\tLL\tLH\tP\tFile\n"
    printf "(LUFS)\t(LUFS)\t(LU)\t(LUFS)\t(LUFS)\t(LUFS)\t(dBFS)\t\n"

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

    integrated=$(extract_from_summary "I" "$summary" "LUFS")
    i_threshold=$(extract_from_summary "Threshold" "$summary" "LUFS" | head -n1)
    lra=$(extract_from_summary "LRA" "$summary" "LU")
    lra_threshold=$(extract_from_summary "Threshold" "$summary" "LUFS" | tail -n1)
    lra_low=$(extract_from_summary "LRA low" "$summary" "LUFS")
    lra_high=$(extract_from_summary "LRA high" "$summary" "LUFS")
    peak=$(extract_from_summary "Peak" "$summary" "dBFS")

    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
        "$integrated" "$i_threshold" \
        "$lra" "${lra_threshold}" "${lra_low}" "${lra_high}" \
        "$peak" "$file"

    return 0
}

extract_from_summary()
{
    local field="$1"
    local summary="$2"
    local unit="$3"

    printf "%s" "$summary" \
        | grep "^ *$field: " \
        | cut -d":" -f2 \
        | sed "s/^ *// ; s/ $unit//"
}

main "$@"
exit $?
