#!/bin/bash

function main()
{
    output_file="$1"

    run_tests 2>&1 | tee "$output_file"
    exit_code=${PIPESTATUS[0]}
    sleep .7
    return $exit_code
}

function run_tests()
{
    if [ -f "Makefile" -o -f "makefile" ]
    then
        bake check
        return $?
    elif [ -f "build.xml" ]
    then
        ant junit
        return $?
    fi
    echo "Where's a makefile or a build.xml when you need it? :-("
    return 0
}

main "$@"
exit $?
