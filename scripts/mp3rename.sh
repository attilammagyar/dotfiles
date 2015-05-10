#!/bin/bash

function main()
{
    local old_name=""
    local new_name=""
    local temp_file=$(tempfile)

    echo "#" >>"$temp_file"
    echo "# review, change or delete the 'new_name < old_name' pairs" >>"$temp_file"
    echo "#" >>"$temp_file"
    echo "" >>"$temp_file"

    while [[ $# -gt 0 ]]
    do
        old_name="$1"
        new_name=$(id_to_name "$old_name")
        if [[ "$new_name" != "$old_name" ]]
        then
            echo "$new_name < $old_name"
        fi
        shift
    done \
        | sort >>"$temp_file"

    "$EDITOR" "$temp_file"

    grep "^[a-z_0-9.-]\\+\\.mp3 < .*\$" "$temp_file" \
        | while read
          do
              new_name=$(echo "$REPLY" | sed "s/^\\([^< ]\\+\\) *< *[^<]*\$/\\1/")
              old_name=$(echo "$REPLY" | sed "s/^[^< ]* *< *\\([^<]*\\)\$/\\1/")

              if [[ ! -e "$old_name" ]]
              then
                  warn "$old_name does not exist"
                  continue
              fi

              if [[ -e "$new_name" ]]
              then
                  warn "$old_name would overwrite $new_name"
                  continue
              fi

              if [[ "$new_name" = "" ]]
              then
                  warn "$old_name would be renamed to empty file name"
                  continue
              fi

              if [[ "$new_name" = "$old_name" ]]
              then
                  warn "$old_name would not be renamed"
                  continue
              fi

              mv -v "$old_name" "$new_name"
          done

    rm "$temp_file"
}

function id_to_name()
{
    local filename="$1"
    local id3=""
    local artist=""
    local album=""
    local track_number=""
    local title=""

    id3=$(id3v2 -l "$filename")
    artist=$(echo "$id3" | extract "^TPE.*: ..*\$" | to_alnum)
    album=$(echo "$id3" | extract "^TALB.*: ..*\$" | to_alnum)
    title=$(echo "$id3" | extract "^TIT.*: ..*\$" | to_alnum)
    track_number=$(
        echo "$id3" \
            | extract "^TRCK.*: [0-9]\\+/[0-9]\\+\$" \
            | cut -d/ -f1 \
            | sed "s/[^0-9]//g ; s/^\\([0-9]\\)\$/0\\1/"
    )

    echo "${artist}-${album}-${track_number}-${title}.mp3"
}

function extract()
{
    local pattern="$1"

    ( grep -m1 "$pattern" | cut -d: -f2- ; echo "unknown" ) | head -n1
}

function to_alnum()
{
    tr -d "'!?" \
        | iconv --from-code UTF-8 --to-code ASCII//TRANSLIT \
        | tr [[:upper:]] [[:lower:]] \
        | sed "s/[^a-zA-Z0-9]\\+/_/g ; s/^_*// ; s/_*\$//"
}

function warn()
{
    local message="$1"

    echo "WARNING: $message" >&2
}

main "$@"



# id3v2 -l 03\ -\ Howling.mp3 
# id3v2 tag info for 03 - Howling.mp3:
# TIT2 (Title/songname/content description): Howling
# TPE1 (Lead performer(s)/Soloist(s)): Abingdon Boys School
# TPE2 (Band/orchestra/accompaniment): Abingdon Boys School
# TALB (Album/Movie/Show title): Teaching Materials
# TYER (Year): 2009
# TRCK (Track number/Position in set): 3/13
# TPOS (Part of a set): 1/1
# TCON (Content type): Rock (17)
# TCOP (Copyright message): (c) 2009 Gan-Shin GmbH
# APIC (Attached picture): ()[, 3]: image/jpeg, 125684 bytes
# PRIV (Private frame):  (unimplemented)
# PRIV (Private frame):  (unimplemented)
# 03 - Howling.mp3: No ID3v1 tag
# 
