# Copyright 2023 MihaiStreames
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


class DownwellError(Exception):
    """Base error for all Downwell project errors."""


class MemoryReadError(DownwellError):
    """OS-level memory read failed."""

    def __init__(self, addr: int, os_error: int | None = None) -> None:
        self.addr = addr
        self.os_error = os_error
        detail = f"(os error {os_error})" if os_error is not None else ""
        super().__init__(f"memory read failed at 0x{addr:x} {detail}")


class FieldResolveError(DownwellError):
    """All pointer chains for a named game field failed."""

    def __init__(self, field: str) -> None:
        self.field = field
        super().__init__(f"could not resolve field {field!r}: all pointer chains failed")
