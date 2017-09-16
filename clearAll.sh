#!/bin/bash
set -e

### All of your tmp data will be saved in ./tmp folder
echo "clear all: rm -r ./tmp"

if [ -d "./tmp" ]; then
	rm ./tmp -r
fi

echo "Done!"

