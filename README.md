# Challenge Format

The repository contains three sandboxes with reference implementations of typical data engineer tasks. These are **reference examples** showing the expected level of complexity, structure, and reproducibility of solutions.

Your goal is **not** to copy or rewrite the code, but to **implement or improve** the approach:
- Reproduce the scenario in your own way
- Optimize the architecture, algorithms, and pipelines
- Extend the functionality (new metrics, checks, features)

All three modules address different aspects:
- `pipeline_anomaly` — ETL and anomaly detection
- `feature_store_ml` — feature store and inference
- `data_quality_monitor` — data quality monitoring

Use them as a reference point, not as a ready-made solution.

> [!TIP]
> Don't forget to leave a Star ⭐

## Directories

| Directory              | Scenario                        | Stack                                 |
|------------------------|---------------------------------|---------------------------------------|
| `pipeline_anomaly`     | Mini-ETL + anomaly detection    | Python 3.11, ClickHouse, scikit-learn |
| `feature_store_ml`     | Feature store and inference     | Python 3.11, LightGBM, ClickHouse     |
| `data_quality_monitor` | Data quality monitoring service | Python 3.11, FastAPI, pydantic        |

Each module contains a `README.md` with a detailed guide on running and infrastructure setup.

## Nx Monorepo

The modules are wrapped in a shared Nx workspace for seamless pipeline management. To pick up dependencies, enable Corepack and pull Yarn 4. Yarn runs in `pnp` mode with a global cache — no `node_modules`:

```bash
corepack enable
corepack prepare yarn@stable --activate
yarn install
```

Run main commands via `yarn nx run <project>:<target>`:

| Project                | Target                    | What it does                              |
|------------------------|---------------------------|-------------------------------------------|
| `pipeline-anomaly`     | `install`                 | Installs poetry dependencies              |
|                        | `infra-up` / `infra-down` | Starts / stops ClickHouse                 |
|                        | `pipeline`                | Runs ETL and anomaly detection            |
|                        | `test`                    | pytest for generators and pipeline        |
| `feature-store-ml`     | `install`                 | Sets up the feature store environment     |
|                        | `infra-up` / `infra-down` | Manages ClickHouse feature marts          |
|                        | `features`                | Materializes features                     |
|                        | `train`                   | Trains the model                          |
|                        | `serve`                   | Uploads predictions                       |
|                        | `test`                    | pytest for registry and pipeline          |
| `data-quality-monitor` | `install`                 | Installs the quality monitoring service   |
|                        | `infra-up` / `infra-down` | Starts ClickHouse + metrics               |
|                        | `api`                     | Runs local FastAPI                        |
|                        | `run-checks`              | Runs quality checks                       |
|                        | `test`                    | pytest for rules                          |

Nx simply calls `make` inside each module, so you can continue using the local Makefiles directly if you don't need Nx.
