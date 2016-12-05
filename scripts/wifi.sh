#!/bin/bash

main()
{
    local pattern='(Cell [0-9]+ - Address:)|(ESSID:)|(Encryption key|IE: WPA)'
    local iface="wlan0"
    local essid=""
    local password=""
    local config_file=""
    local error=0

    # service network-manager stop

    ifconfig "$iface" up || exit 1

    echo "Scanning available networks:"
    iwlist "$iface" scan | grep --color=never -E "$pattern"

    echo -n "Enter ESSID: "
    read
    essid="$REPLY"

    echo -n "Enter password: "
    stty -echo
    read
    password="$REPLY"
    stty echo

    echo
    echo "Connecting to '$essid'"

    #iwconfig "$iface" essid "$essid" key "s:$key" || exit 2

    config_file="`tempfile`"

    # echo "ap_scan=2" >"$config_file"
    wpa_passphrase "$essid" "$password" >>"$config_file"

    password=""
    essid=""

    wpa_supplicant -B -D nl80211 -i "$iface" -c "$config_file"
    # wpa_supplicant -B -D wext -i "$iface" -c "$config_file"

    error=$?
    rm "$config_file"

    [ $error -eq 0 ] || exit 2

    echo "Obtaining IP address"

    dhclient -v "$iface" || exit 4
}

main "$@"
