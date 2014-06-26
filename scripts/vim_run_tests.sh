#!/bin/bash

function main()
{
    local currently_edited_file="$1"
    local output_file="$2"

    run_tests "$currently_edited_file" 2>&1 | tee "$output_file"
    local exit_code=${PIPESTATUS[0]}
    sleep .7

    return $exit_code
}

function run_tests()
{
    local currently_edited_file="$1"

    if [[ -f ".vim_run_tests.sh" ]]
    then
        "$SHELL" .vim_run_tests.sh "$currently_edited_file"
        return $?
    elif [[ -f "Makefile" ]] || [[ -f "makefile" ]]
    then
        bake check
        return $?
    elif [[ -f "build.xml" ]]
    then
        ant junit
        return $?
    fi

    echo "ERROR: Create a .vim_run_tests.sh, a Makefile" \
         " or a build.xml first!" >&2

    return 0
}

main "$@"
exit $?
