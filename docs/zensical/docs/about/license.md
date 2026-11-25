---
title: License
description: Licensing terms for the Gosmart.Health Dictionary Service.
---

# License

Copyright (c) 2026, GoSmart Health

Licensed under the **Apache License, Version 2.0** (the "License"); you may not
use this project except in compliance with the License. You may obtain a copy of
the License at:

<https://www.apache.org/licenses/LICENSE-2.0>

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the License
for the specific language governing permissions and limitations under the
License.

## Third-party components

This service builds on open-source projects, each under its own license:

| Component                                    | License     |
| -------------------------------------------- | ----------- |
| [FastAPI](https://fastapi.tiangolo.com/)     | MIT         |
| [Uvicorn](https://www.uvicorn.org/)          | BSD-3-Clause |
| [Genkit](https://python.api.genkit.dev/)     | Apache-2.0  |
| [Neo4j Python driver](https://github.com/neo4j/neo4j-python-driver) | Apache-2.0 |
| [Zensical](https://zensical.org/)            | MIT         |

A full software bill of materials can be generated with
`uv run cyclonedx-bom` once that tool is added to the `dev` dependency group.
