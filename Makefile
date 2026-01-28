# Sequence Map Generator - Development Commands

.PHONY: install dev backend frontend build clean

# Install all dependencies
install:
	@echo "Installing backend dependencies..."
	cd api && pip install -r requirements.txt
	@echo "Installing frontend dependencies..."
	cd frontend && npm install
	@echo "Done!"

# Run both backend and frontend
dev:
	./scripts/run-dev.sh

# Run backend only
backend:
	cd api && python server.py

# Run frontend only
frontend:
	cd frontend && npm run dev

# Build frontend for production
build:
	cd frontend && npm run build

# Clean build artifacts
clean:
	rm -rf frontend/dist
	rm -rf frontend/node_modules/.vite
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Type check frontend
typecheck:
	cd frontend && npm run type-check

# Lint frontend
lint:
	cd frontend && npm run lint
