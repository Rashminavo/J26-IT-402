# J26-IT-402 Backend

## Purpose

The backend provides the service boundary between the research components,
shared score interface, future fusion engine and application interfaces.

## Technology

- Python
- Flask
- JSON Schema

## Current Responsibilities

- Health check endpoint
- Component score submission
- Component score validation
- Temporary in-memory score storage
- Component score retrieval

## API

### Health

`GET /health`

### Submit Component Score

`POST /api/v1/scores`

### Get All Scores

`GET /api/v1/scores`

### Get One Score

`GET /api/v1/scores/{observation_id}`

## Current Storage

Scores are currently stored in memory for the backend MVP.

Persistent database storage is intentionally deferred until the API contract
and component integration requirements are stable.

## Component Boundary

The backend does not contain the internal ML implementation of the four
research components.

Each component remains independently developed and exposes structured outputs
through the shared score contract.