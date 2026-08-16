#!/usr/bin/env bash

set -euo pipefail

# Modify these constants before each download. This script accepts no arguments.
# Connection settings (LFTP_HOST, LFTP_USER, LFTP_PASSWORD, SSL_CA_FILE) are
# loaded from the repository .env file, which is gitignored.
REMOTE_VIDEO_PATH="20260810/4kvideo/2026-08-10_14-32-37/_video"
START_VIEW=49
END_VIEW=80

if (( $# != 0 )); then
    echo "This script accepts no arguments. Modify the constants at the top instead." >&2
    exit 2
fi

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
environment_file="$repository_root/.env"
if [[ ! -r "$environment_file" ]]; then
    echo "Cannot read $environment_file." >&2
    exit 1
fi

set -a
# shellcheck disable=SC1090
source "$environment_file"
set +a

for connection_variable in LFTP_HOST LFTP_USER LFTP_PASSWORD SSL_CA_FILE; do
    if [[ -z "${!connection_variable:-}" ]]; then
        echo "$connection_variable is not defined or empty in $environment_file." >&2
        exit 1
    fi
done

if ! command -v lftp >/dev/null 2>&1; then
    echo "lftp is not installed or is not available in PATH." >&2
    exit 1
fi

if [[ ! -r "$SSL_CA_FILE" ]]; then
    echo "SSL CA file is not readable: $SSL_CA_FILE" >&2
    exit 1
fi

if [[ ! "$START_VIEW" =~ ^[0-9]+$ || ! "$END_VIEW" =~ ^[0-9]+$ ]]; then
    echo "START_VIEW and END_VIEW must be non-negative integers." >&2
    exit 1
fi

if (( START_VIEW > END_VIEW )); then
    echo "START_VIEW must not be greater than END_VIEW." >&2
    exit 1
fi

remote_video_path="${REMOTE_VIDEO_PATH#/}"
remote_video_path="${remote_video_path%/}"
remote_date_directory="${remote_video_path%%/*}"
video_directory_name="${remote_video_path##*/}"
capture_directory="$(basename "$(dirname "$remote_video_path")")"

if [[ "$remote_video_path" == "$remote_date_directory" ]]; then
    echo "REMOTE_VIDEO_PATH must include the date directory and the _video directory." >&2
    exit 1
fi

if [[ ! "$remote_date_directory" =~ ^[0-9]{8}$ ]]; then
    echo "The first REMOTE_VIDEO_PATH component must be an eight-digit date such as 20260810." >&2
    exit 1
fi

if [[ "$video_directory_name" != "_video" ]]; then
    echo "REMOTE_VIDEO_PATH must end with _video." >&2
    exit 1
fi

if [[ ! "$capture_directory" =~ ^[0-9]{4}-[0-9]{2}-[0-9]{2}_[0-9]{2}-[0-9]{2}-[0-9]{2}$ ]]; then
    echo "The parent of _video must be a timestamp such as 2026-08-10_14-32-37." >&2
    exit 1
fi

date_identifier="${remote_date_directory:4:4}"
local_date_directory="$repository_root/data/$date_identifier"
local_capture_directory="$local_date_directory/$capture_directory"
local_video_directory="$local_capture_directory/_video"

mkdir -p "$local_date_directory"

echo "Remote videos:     $remote_video_path/cam_{$START_VIEW..$END_VIEW}.mp4"
echo "Local videos:      $local_video_directory"
echo

print_lftp_settings() {
    printf '%s\n' \
        'set ssl:check-hostname false' \
        "set ssl:ca-file $SSL_CA_FILE" \
        'set net:max-retries 5' \
        'set net:timeout 30' \
        'set net:reconnect-interval-base 5' \
        'set net:reconnect-interval-max 30' \
        'set cmd:fail-exit true'
}

echo "Finding calibration file in $remote_date_directory..."
calibration_listing="$({
    print_lftp_settings
    printf '%s\n' \
        "cd $remote_date_directory" \
        'cls -1 -s --block-size=1 --filesize calibration*' \
        'bye'
} | lftp -u "$LFTP_USER,$LFTP_PASSWORD" "$LFTP_HOST")"

mapfile -t calibration_lines < <(
    printf '%s\n' "$calibration_listing" | sed '/^[[:space:]]*$/d; s/\r$//'
)

if (( ${#calibration_lines[@]} != 1 )); then
    echo "Expected exactly one calibration* file in $remote_date_directory, found ${#calibration_lines[@]}." >&2
    if (( ${#calibration_lines[@]} > 0 )); then
        printf '  %s\n' "${calibration_lines[@]}" >&2
    fi
    exit 1
fi

if [[ ! "${calibration_lines[0]}" =~ ^[[:space:]]*([0-9]+)[[:space:]]+(.+)$ ]]; then
    echo "Could not parse calibration file size: ${calibration_lines[0]}" >&2
    exit 1
fi

calibration_remote_size="${BASH_REMATCH[1]}"
calibration_filename="${BASH_REMATCH[2]}"
if [[ "$calibration_filename" == */* ]]; then
    echo "Unexpected calibration path returned by the server: $calibration_filename" >&2
    exit 1
fi

local_calibration_path="$local_date_directory/$calibration_filename"
echo "Remote calibration: $remote_date_directory/$calibration_filename"
echo "Local calibration:  $local_calibration_path"
echo

calibration_local_size=0
calibration_exists=false
if [[ -e "$local_calibration_path" ]]; then
    if [[ ! -f "$local_calibration_path" ]]; then
        echo "Local calibration path is not a regular file: $local_calibration_path" >&2
        exit 1
    fi
    calibration_exists=true
    calibration_local_size="$(stat -c '%s' "$local_calibration_path")"
fi

if (( calibration_local_size > calibration_remote_size )); then
    echo "Local calibration is larger than the remote file; refusing to overwrite it." >&2
    echo "Local: $calibration_local_size bytes, remote: $calibration_remote_size bytes" >&2
    exit 1
elif [[ "$calibration_exists" == true ]] && (( calibration_local_size == calibration_remote_size )); then
    echo "Skipping calibration: already complete ($calibration_remote_size bytes)."
else
    if (( calibration_local_size == 0 )); then
        echo "Downloading calibration ($calibration_remote_size bytes)..."
    else
        echo "Resuming calibration at $calibration_local_size/$calibration_remote_size bytes..."
    fi

    {
        print_lftp_settings
        printf '%s\n' \
            "cd $remote_date_directory" \
            "lcd $local_date_directory" \
            "get -c \"$calibration_filename\"" \
            'bye'
    } | lftp -u "$LFTP_USER,$LFTP_PASSWORD" "$LFTP_HOST"

    calibration_local_size="$(stat -c '%s' "$local_calibration_path")"
    if (( calibration_local_size != calibration_remote_size )); then
        echo "Calibration size verification failed after download." >&2
        echo "Local: $calibration_local_size bytes, remote: $calibration_remote_size bytes" >&2
        exit 1
    fi
fi

mkdir -p "$local_video_directory"

echo
echo "Reading remote video sizes..."
video_listing="$({
    print_lftp_settings
    printf '%s\n' \
        "cd /$remote_video_path" \
        'cls -1 -s --block-size=1 --filesize cam_*.mp4' \
        'bye'
} | lftp -u "$LFTP_USER,$LFTP_PASSWORD" "$LFTP_HOST")"

declare -A remote_video_sizes=()
while IFS= read -r video_line; do
    video_line="${video_line%$'\r'}"
    [[ -z "$video_line" ]] && continue

    if [[ ! "$video_line" =~ ^[[:space:]]*([0-9]+)[[:space:]]+cam_([0-9]+)\.mp4$ ]]; then
        echo "Could not parse remote video size: $video_line" >&2
        exit 1
    fi

    remote_video_sizes["${BASH_REMATCH[2]}"]="${BASH_REMATCH[1]}"
done <<< "$video_listing"

total_views=$((END_VIEW - START_VIEW + 1))
download_views=()
skipped_views=0

for ((view = START_VIEW; view <= END_VIEW; view++)); do
    filename="cam_$view.mp4"
    local_video_path="$local_video_directory/$filename"
    remote_size="${remote_video_sizes[$view]:-}"

    if [[ -z "$remote_size" ]]; then
        echo "Remote video is missing: $remote_video_path/$filename" >&2
        exit 1
    fi

    local_size=0
    local_exists=false
    if [[ -e "$local_video_path" ]]; then
        if [[ ! -f "$local_video_path" ]]; then
            echo "Local video path is not a regular file: $local_video_path" >&2
            exit 1
        fi
        local_exists=true
        local_size="$(stat -c '%s' "$local_video_path")"
    fi

    if (( local_size > remote_size )); then
        echo "Local video is larger than the remote file: $local_video_path" >&2
        echo "Local: $local_size bytes, remote: $remote_size bytes" >&2
        exit 1
    elif [[ "$local_exists" == true ]] && (( local_size == remote_size )); then
        position=$((view - START_VIEW + 1))
        echo "[$position/$total_views] Skipping $filename: already complete ($remote_size bytes)."
        skipped_views=$((skipped_views + 1))
    else
        download_views+=("$view")
    fi
done

if (( ${#download_views[@]} > 0 )); then
    echo
    echo "Downloading incomplete videos one at a time..."
    {
        print_lftp_settings
        printf '%s\n' \
            "cd /$remote_video_path" \
            "lcd $local_video_directory"

        for view in "${download_views[@]}"; do
            position=$((view - START_VIEW + 1))
            local_size=0
            local_video_path="$local_video_directory/cam_$view.mp4"
            if [[ -f "$local_video_path" ]]; then
                local_size="$(stat -c '%s' "$local_video_path")"
            fi

            if (( local_size == 0 )); then
                printf 'echo [%d/%d] Downloading cam_%d.mp4\n' "$position" "$total_views" "$view"
            else
                printf 'echo [%d/%d] Resuming cam_%d.mp4 at %d bytes\n' \
                    "$position" "$total_views" "$view" "$local_size"
            fi
            printf 'get -c cam_%d.mp4\n' "$view"
        done

        printf '%s\n' 'bye'
    } | lftp -u "$LFTP_USER,$LFTP_PASSWORD" "$LFTP_HOST"
else
    echo "All requested videos are already complete; no video download connection is needed."
fi

for ((view = START_VIEW; view <= END_VIEW; view++)); do
    local_video_path="$local_video_directory/cam_$view.mp4"
    local_size="$(stat -c '%s' "$local_video_path")"
    remote_size="${remote_video_sizes[$view]}"
    if (( local_size != remote_size )); then
        echo "Video size verification failed: $local_video_path" >&2
        echo "Local: $local_size bytes, remote: $remote_size bytes" >&2
        exit 1
    fi
done

echo
echo "Download completed: $local_capture_directory"
echo "Skipped complete videos: $skipped_views"
echo "Downloaded or resumed videos: ${#download_views[@]}"
