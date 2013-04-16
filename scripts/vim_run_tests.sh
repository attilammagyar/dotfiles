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
    if [ -f ".vim_run_tests.sh" ]
    then
        source .vim_run_tests.sh
        return $?
    elif [ -f "Makefile" -o -f "makefile" ]
    then
        bake check
        return $?
    elif [ -f "build.xml" ]
    then
        ant junit
        return $?
    fi
    echo "ERROR: Create a .vim_run_tests.sh, a Makefile or a build.xml first!" >&2
    return 0
}

main "$@"
exit $?
