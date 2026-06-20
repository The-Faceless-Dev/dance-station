# Make Runtime UV Sync Container-Safe Plan

## Problem

`autotransition setup` can fail inside containers or mounted workspaces when `uv sync` tries to hardlink/copy package files into the ACE-Step runtime `.venv`. The observed failure is a stale file handle while installing `transformers`, after uv already warned that hardlinking was not supported.

## Approach

- Run ACE-Step `uv sync` with `UV_LINK_MODE=copy` by default.
- Preserve existing environment variables and only fill the value when users have not already set it.
- Keep the setup command unchanged for users.

## Risk

Copy mode can use more disk and may be slower, but it is more reliable across Docker, network mounts, and overlay filesystems.
