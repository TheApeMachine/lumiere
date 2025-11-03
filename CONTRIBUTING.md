# Contributing to Lumiere

Thank you for your interest in contributing to Lumiere! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful, professional, and inclusive. We welcome contributions from everyone.

## Getting Started

### Prerequisites

- Go 1.21 or higher
- Git
- Basic understanding of Go and REST APIs

### Setting Up Development Environment

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/YOUR_USERNAME/lumiere.git
   cd lumiere
   ```
3. Install dependencies:
   ```bash
   go mod download
   ```
4. Build the project:
   ```bash
   go build -o lumiere .
   ```
5. Run tests:
   ```bash
   go test ./...
   ```

## Development Workflow

### 1. Create a Branch

```bash
git checkout -b feature/your-feature-name
```

Use descriptive branch names:
- `feature/` for new features
- `fix/` for bug fixes
- `docs/` for documentation
- `refactor/` for code improvements

### 2. Make Changes

- Write clean, readable code
- Follow Go best practices and idioms
- Add comments for complex logic
- Keep functions small and focused
- Use meaningful variable names

### 3. Write Tests

- Add unit tests for new functions
- Ensure existing tests still pass
- Aim for good test coverage
- Test edge cases and error conditions

```bash
go test ./... -v
```

### 4. Run Linting

Format your code:
```bash
go fmt ./...
```

Run static analysis:
```bash
go vet ./...
```

### 5. Update Documentation

- Update README.md if adding features
- Add/update code comments
- Update ARCHITECTURE.md for architectural changes
- Include examples in comments

### 6. Commit Changes

Write clear commit messages:
```
<type>: <subject>

<body>

<footer>
```

Types:
- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation only
- `refactor:` Code restructuring
- `test:` Adding tests
- `chore:` Maintenance

Example:
```
feat: add support for video style presets

- Add StylePreset model
- Implement preset selection in API
- Add default preset collection
- Update documentation

Closes #123
```

### 7. Push and Create Pull Request

```bash
git push origin feature/your-feature-name
```

Create a Pull Request with:
- Clear title and description
- Reference related issues
- List of changes made
- Screenshots (if UI changes)

## Coding Standards

### Go Style Guide

Follow the official [Go Code Review Comments](https://github.com/golang/go/wiki/CodeReviewComments).

### Project-Specific Guidelines

1. **Error Handling**
   ```go
   if err != nil {
       return fmt.Errorf("failed to process: %w", err)
   }
   ```

2. **Configuration**
   - Use environment variables
   - Provide sensible defaults
   - Document all config options

3. **Logging**
   ```go
   log.Printf("Processing project %s", projectID)
   ```

4. **File Permissions**
   - Use `config.DefaultFilePerms` for files
   - Use `config.DefaultDirPerms` for directories

5. **Concurrency**
   - Use goroutines for long-running tasks
   - Protect shared state with mutexes
   - Use channels for communication

### Package Organization

```
- `main.go` - Application entry point
- `config/` - Configuration management
- `api/` - HTTP handlers and routing
- `models/` - Data structures
- `pipeline/` - Core processing logic
```

## Testing Guidelines

### Unit Tests

Test individual functions:
```go
func TestGenerateConcept(t *testing.T) {
    cg := NewConceptGenerator()
    concept, err := cg.Generate("/path/to/audio.mp3", "test prompt")
    
    if err != nil {
        t.Fatalf("Generate failed: %v", err)
    }
    
    if len(concept.KeyMoments) != 7 {
        t.Errorf("Expected 7 key moments, got %d", len(concept.KeyMoments))
    }
}
```

### Integration Tests

Test component interactions (in separate test files):
```go
func TestProjectWorkflow(t *testing.T) {
    // Test full pipeline execution
}
```

### Test Coverage

Run with coverage:
```bash
go test ./... -cover
```

Generate coverage report:
```bash
go test ./... -coverprofile=coverage.out
go tool cover -html=coverage.out
```

## Pull Request Process

1. **Before Submitting**
   - [ ] All tests pass
   - [ ] Code is formatted (`go fmt`)
   - [ ] No linting errors (`go vet`)
   - [ ] Documentation updated
   - [ ] Commit messages are clear

2. **PR Description Should Include**
   - What changes were made
   - Why the changes were needed
   - How to test the changes
   - Any breaking changes
   - Related issue numbers

3. **Review Process**
   - Maintainers will review your PR
   - Address any feedback
   - Keep PR focused and small
   - Be responsive to comments

4. **After Approval**
   - Maintainer will merge your PR
   - Your contribution will be in the next release

## Feature Requests

To request a feature:

1. Check existing issues first
2. Create a new issue with:
   - Clear description of the feature
   - Use cases and benefits
   - Possible implementation approach
   - Willingness to contribute

## Bug Reports

To report a bug:

1. Check if it's already reported
2. Create a new issue with:
   - Clear description
   - Steps to reproduce
   - Expected behavior
   - Actual behavior
   - Environment (OS, Go version)
   - Logs/screenshots if applicable

## Areas for Contribution

We especially welcome contributions in these areas:

### High Priority
- [ ] Integration with actual AI APIs (Stable Diffusion, DALL-E, etc.)
- [ ] Audio analysis with beat detection
- [ ] Video composition with FFmpeg
- [ ] Database integration for project persistence
- [ ] WebSocket support for real-time updates

### Medium Priority
- [ ] Web UI for easier interaction
- [ ] Additional visual styles and presets
- [ ] Batch processing capabilities
- [ ] Preview generation
- [ ] Caching layer for faster regeneration

### Documentation
- [ ] More usage examples
- [ ] Video tutorials
- [ ] API client libraries (Python, JavaScript)
- [ ] Architecture diagrams
- [ ] Performance optimization guide

### Testing
- [ ] More comprehensive test coverage
- [ ] Integration test suite
- [ ] Load testing
- [ ] Benchmark tests

## Questions?

- Open an issue with the `question` label
- Join discussions in existing issues
- Check the [ARCHITECTURE.md](ARCHITECTURE.md) for technical details

## Recognition

Contributors will be recognized in:
- README.md contributors section
- Release notes
- Project documentation

Thank you for contributing to Lumiere! 🎬✨
