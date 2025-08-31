#!/bin/bash
# Enhanced development environment setup script
# Prevents formatting issues from reaching remote repos

set -e

echo "🚀 Setting up development environment with enhanced quality gates..."

# Install dependencies
echo "📦 Installing dependencies..."
pip install -e ".[dev]"

# Install pre-commit hooks (mandatory)
echo "🪝 Installing pre-commit hooks..."
pre-commit install

# Install pre-push hook to prevent unformatted code from reaching remote
echo "🛡️  Installing pre-push formatting guard..."
cat > .git/hooks/pre-push << 'EOF'
#!/bin/bash
# Pre-push hook: Run formatting checks before push
echo "🔍 Running pre-push formatting checks..."
pre-commit run --all-files --show-diff-on-failure

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ PUSH BLOCKED: Formatting checks failed"
    echo "💡 Fix formatting issues with: pre-commit run --all-files"
    echo "💡 Then try pushing again"
    echo ""
    exit 1
fi

echo "✅ All formatting checks passed - push allowed"
EOF
chmod +x .git/hooks/pre-push

# Create convenience scripts
echo "🛠️  Creating convenience scripts..."

# Format-check script
cat > format-check.sh << 'EOF'
#!/bin/bash
echo "🔍 Running all formatting and quality checks..."
echo ""

# Run all pre-commit hooks
pre-commit run --all-files --show-diff-on-failure

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ All checks passed! Code is ready for commit/push."
else
    echo ""
    echo "❌ Some checks failed. Fix the issues above before committing."
    exit 1
fi
EOF
chmod +x format-check.sh

# Safe commit script
cat > safe-commit.sh << 'EOF'
#!/bin/bash
if [ $# -eq 0 ]; then
    echo "Usage: ./safe-commit.sh \"commit message\""
    exit 1
fi

echo "🔍 Running pre-commit checks before commit..."
pre-commit run --all-files

if [ $? -eq 0 ]; then
    echo "✅ All checks passed - proceeding with commit"
    git commit -m "$1"
else
    echo "❌ Pre-commit checks failed - commit blocked"
    exit 1
fi
EOF
chmod +x safe-commit.sh

# Safe push script
cat > safe-push.sh << 'EOF'
#!/bin/bash
echo "🔍 Running final checks before push..."
pre-commit run --all-files

if [ $? -eq 0 ]; then
    echo "✅ All checks passed - proceeding with push"
    git push "$@"
else
    echo "❌ Pre-commit checks failed - push blocked"
    exit 1
fi
EOF
chmod +x safe-push.sh

# Setup git aliases for easy use
echo "⚙️  Setting up helpful git aliases..."
git config alias.cfmt '!pre-commit run --all-files'
git config alias.check '!./format-check.sh'
git config alias.safe-commit '!./safe-commit.sh'
git config alias.safe-push '!./safe-push.sh'

# Create .githooks directory for future use
mkdir -p .githooks

echo ""
echo "✅ Development environment setup complete!"
echo ""
echo "📋 Available commands:"
echo "  ./format-check.sh         - Run all formatting checks"
echo "  ./safe-commit.sh \"msg\"    - Format check + commit"
echo "  ./safe-push.sh            - Format check + push"
echo ""
echo "📋 Git aliases:"
echo "  git cfmt                  - Run formatting checks"
echo "  git check                 - Run all checks"
echo "  git safe-commit \"msg\"     - Safe commit with checks"
echo "  git safe-push             - Safe push with checks"
echo ""
echo "🛡️  Pre-push hook installed - formatting issues will be blocked before reaching remote"
echo "💡 Use the safe-* commands to avoid formatting issues entirely"
echo ""
