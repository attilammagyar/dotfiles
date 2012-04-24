#!/bin/bash

if [ "x$1" = "x" -o "x$2" = "x" ]
then
	echo "Usage: $0 backup_file restore_directory" >&2
	exit 1
fi

backup_file="$1"
restore_dir="$2"

if [ ! -f "$backup_file" ]
then
	echo "$backup_file is not a file" >&2
	exit 2
fi
if [ ! -d "$restore_dir" ]
then
	echo "$restore_dir is not a directory" >&2
	exit 3
fi

echo -n "Decryption password: "
stty -echo
read
password="$REPLY"
stty echo
REPLY=""

echo -e "\nRestoring files from $backup_file"

abspath=$(cd $(dirname "$backup_file") ; pwd)"/"$(basename "$backup_file") 
cd "$restore_dir"
dd if="$abspath" | openssl aes-256-ecb -d -k "$password" | tar -xzvf - && echo "OK"
cd -

