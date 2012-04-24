#!/bin/bash

if [ "x$1" = "x" -o "x$2" = "x" ]
then
	echo "Usage: $0 directory backup_path" >&2
	exit 1
fi

directory="$1"
backup_path="$2"

if [ ! -d "$directory" ]
then
	echo "$directory is not a directory" >&2
	exit 2
fi
if [ ! -d "$backup_path" ]
then
	echo "$backup_path is not a directory" >&2
	exit 3
fi

echo -n "Encryption password: "
stty -echo
read
password="$REPLY"
stty echo
REPLY=""

echo -en "\nConfirm: "
stty -echo
read
confirm="$REPLY"
stty echo
REPLY=""

if [ "x$confirm" != "x$password" ]
then
	echo -e "\nPassword and confirmation does not match" >&2
	exit 4
fi

confirm=""

backup_name="$backup_path/"$(basename "$directory")"-"$(date '+%Y-%m-%d-%H-%M-%S')

echo -e "\nBackup will be created to $backup_name.tgz.aes256"

tar -czvf - --index-file="$backup_name.index" "$directory/" | openssl aes-256-ecb -salt -k "$password" | dd of="$backup_name.tgz.aes256" && echo "OK"

