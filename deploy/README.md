# Deploying the Seldon MCP server

How `https://mcp.neuralk.ai` is run, and how to run another one. The chart is
[`helm/seldon-mcp`](helm/seldon-mcp); the Neuralk production values are
[`helm/seldon-mcp/values-neuralk.yaml`](helm/seldon-mcp/values-neuralk.yaml).

## What it is

One stateless process behind the cluster's nginx ingress. It holds **no Neuralk
key**: every MCP request carries the client's own key
(`REQUIRE_CLIENT_API_KEY=true`), the server validates it against the prediction
API and forwards the work. A request without a key is answered `401` before it
reaches a tool. The server writes one kind of file — the single-use prediction
CSVs behind `/downloads/<token>` — into an emptyDir that lives five minutes.

Because those files are pod-local, the chart runs **one replica** and refuses
to render more unless `downloads.existingClaim` names a ReadWriteMany volume
every pod can see. Rollouts are gapless anyway (`maxSurge: 1`,
`maxUnavailable: 0`).

## The Neuralk deployment, step by step

Everything below was done once on 2026-09-02 and is recorded here so the next
person can redo it, on this cluster or another.

### 1. The image

`.github/workflows/image.yml` builds `rg.fr-par.scw.cloud/neuralk-prod/seldon-mcp`
on every push to `main` that touches the server, tagged `main-<sha>`. It needs
one repository secret, `SCW_REGISTRY_SECRET_KEY`: the secret key of the
Scaleway IAM application **`seldon-mcp-ci`**, whose only policy is
`ContainerRegistryFullAccess` on the Production project.

To build by hand (Apple Silicon: the nodes are x86, build for them):

```bash
docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  -t rg.fr-par.scw.cloud/neuralk-prod/seldon-mcp:main-$(git rev-parse --short HEAD) --push .
```

### 2. The namespace and the pull secret

The registry is private. The pull secret carries the key of the IAM application
**`seldon-mcp-pull`** (`ContainerRegistryReadOnly` on Production — it can pull
and do nothing else):

```bash
NS=seldon-mcp
kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f -
kubectl -n "$NS" create secret docker-registry scaleway-registry \
  --docker-server=rg.fr-par.scw.cloud \
  --docker-username=nologin \
  --docker-password='<seldon-mcp-pull secret key>' \
  --dry-run=client -o yaml | kubectl apply -f -
```

### 3. DNS

`mcp.neuralk.ai` is an `A` record on Cloudflare, DNS-only (grey cloud), TTL 300,
pointing at the ingress load balancer — the same address as every other
`*.neuralk.ai` product of the cluster:

```bash
kubectl -n ingress-nginx get svc ingress-nginx-controller \
  -o jsonpath='{.status.loadBalancer.ingress[0].ip}'
```

The certificate is issued by the `letsencrypt-prod-cloudflare` ClusterIssuer
(DNS-01), so the record must exist before the release is installed or the
challenge waits on it.

### 4. Install

Pin the image tag in `values-neuralk.yaml`, then:

```bash
helm upgrade --install seldon-mcp deploy/helm/seldon-mcp \
  -n seldon-mcp -f deploy/helm/seldon-mcp/values-neuralk.yaml
kubectl -n seldon-mcp get certificate,pods,ingress
```

### 5. Verify from outside

```bash
curl -fsS https://mcp.neuralk.ai/healthz                       # ok
curl -s -o /dev/null -w '%{http_code}\n' -X POST https://mcp.neuralk.ai/mcp   # 401: no key
# a real MCP handshake, with a key
curl -s https://mcp.neuralk.ai/mcp \
  -H "x-neuralk-api-key: $NEURALK_API_KEY" \
  -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-06-18","capabilities":{},"clientInfo":{"name":"curl","version":"0"}}}'
```

Then, from a client: `claude mcp add --transport http seldon https://mcp.neuralk.ai/mcp --header "x-neuralk-api-key: $NEURALK_API_KEY"`
and call `list_models`.

## Upgrading

1. Merge to `main`; the Image workflow pushes `main-<sha>`.
2. Set `image.tag` in `values-neuralk.yaml` to that tag, in the same change as
   anything else the release needs.
3. `helm upgrade` as above. A rollback is the previous tag and the same command.

## What can go wrong

- **`421` or "Invalid Host header"** from the pod: the process was started on a
  loopback host with FastMCP's DNS-rebinding protection on. The chart binds
  `0.0.0.0`, which turns it off; do not change `--host`.
- **Download links `404` after a scale-up**: two replicas, pod-local files.
  Scale back to one or name a shared claim.
- **Tool calls cut at 60 s**: the ingress timeout annotations were dropped from
  values. The chart's default is 3600 s.
- **`401` for a key that works in the SDK**: the key check goes to
  `NEURALK_PREDICTION_URL/api/v1/auth/whoami`; check that URL from the pod.
