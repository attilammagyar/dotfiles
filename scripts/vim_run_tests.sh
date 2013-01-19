#!/bin/bash

run_tests()
{
    if [ -f "Makefile" -o -f "makefile" ]
    then
        bake check
        return $?
    elif [ -f "build.xml" ]
    then
        ant test || ant junit
        return $?
    fi
    echo "Where's a makefile or a build.xml when you need it? :-("
    return 0
}

output_file="$1"
if [ "$output_file" = "" ]
then
    output_file="/dev/null"
fi

cd "`git rev-parse --show-cdup`" 2>&1 >/dev/null
run_tests 2>&1 | tee "$output_file"
exit_code=${PIPESTATUS[0]}
sleep .7
exit $exit_code
