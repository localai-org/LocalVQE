if(NOT URL OR NOT OUTFILE OR NOT EXPECTED_SHA256)
    message(FATAL_ERROR "download_asset.cmake requires -DURL, -DOUTFILE, -DEXPECTED_SHA256")
endif()

# file(DOWNLOAD ... EXPECTED_HASH ...) short-circuits when OUTFILE already
# matches the hash (CMake ≥ 3.7), so no manual exists/hash pre-check needed.
file(DOWNLOAD "${URL}" "${OUTFILE}"
    SHOW_PROGRESS
    STATUS _status
    EXPECTED_HASH SHA256=${EXPECTED_SHA256})
list(GET _status 0 _code)
list(GET _status 1 _msg)
if(NOT _code EQUAL 0)
    file(REMOVE "${OUTFILE}")
    message(FATAL_ERROR "Download failed (${_code}): ${_msg}")
endif()
