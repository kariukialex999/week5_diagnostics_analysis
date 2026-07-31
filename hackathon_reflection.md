# Hackathon #1 Reflection

## Project: Sentinel — HSE Early Warning Detection System

### Biggest Technical Hurdle

The biggest technical hurdle our team faced was bridging the Python ETL pipeline with the Spring Boot backend in a way that worked both locally and in production. Locally, both processes share the same filesystem, so the ETL writes `live_batch.json` and the backend reads it. In production, however, the Python runtime and the Java service run on completely different machines, making the file-based handoff impossible. We had to rethink the integration mid-way through the hackathon and implement an HTTP push mechanism — a new `POST /api/etl/push` endpoint on the backend that accepts the batch payload directly — while keeping the local file-based flow intact for development.

### How We Resolved It

We extended `run_pipeline.py` with a `--push-url` flag and used `urllib` to POST the generated batch to the backend API, secured with a pre-shared API key passed via an `X-ETL-Api-Key` header. The backend delegates the payload to the existing loading logic, keeping duplication minimal.

### What We Would Do Differently

We would establish API contracts between services earlier, before writing any implementation code. Mismatched assumptions between the Python and Java sides cost us significant debugging time. Clearer interface definitions and earlier integration testing would have surfaced the filesystem coupling issue before it became a blocker.
