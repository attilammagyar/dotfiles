#!/bin/bash

service network-manager stop

ifconfig wlan0 up || exit 1

echo "Scanning available networks:"
pattern='(Cell [0-9]+ - Address:)|(ESSID:)|(Encryption key|IE: WPA)'
iwlist wlan0 scan | grep --color=never -E "$pattern"

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

#iwconfig wlan0 essid "$essid" key "s:$key" || exit 2

config_file="`tempfile`"

wpa_passphrase "$essid" "$password" >"$config_file"

password=""
essid=""

wpa_supplicant -B -D nl80211 -i wlan0 -c "$config_file"

error=$?
rm "$config_file"

[ $error -eq 0 ] || exit 2

echo "Obtaining IP address"

dhclient wlan0 || exit 3
