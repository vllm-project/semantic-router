/*
Copyright 2025 vLLM Semantic Router.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

package valkey

// ClientInfoTag is the value every Valkey GLIDE client built by
// semantic-router sets via WithClientInfoTag. The server surfaces it as
// lib-name=GlideGo(semantic-router) in CLIENT INFO / CLIENT LIST, visible
// only to whoever runs those commands against the operator's own Valkey
// deployment - it is never sent in any response or payload to an inference
// caller.
//
// One constant shared by the cache, vector store and agentic memory
// constructors, so a rename can't let the three drift out of sync.
const ClientInfoTag = "semantic-router"
