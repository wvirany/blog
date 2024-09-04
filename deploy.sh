#!/bin/bash

function push_blog {
    git add . && git commit -m "Update public" && git push
}

# function rebuild_public {
#     clean_public
#     hugo --minify
# }

# function clean_public {
#     rm -r public/*
# }

function push_public {
    cd public && git add . && git commit -m "Site update" && git push
}

function all {
    push_public
    push_blog
}

function dev {
    hugo server --disableFastRender -D
}

# Call the function passed as the first argument
$1
