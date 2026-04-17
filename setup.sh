#!/bin/bash

# Fake Review Detector - Setup Script
# Automates the setup process for the project

set -e

echo "=========================================="
echo "Fake Review Detector - Setup Script"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check Python version
check_python() {
    print_status "Checking Python version..."
    
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
        print_status "Found Python $PYTHON_VERSION"
        
        # Check if version is 3.8 or higher
        REQUIRED_VERSION="3.8"
        if [ "$(printf '%s\n' "$REQUIRED_VERSION" "$PYTHON_VERSION" | sort -V | head -n1)" = "$REQUIRED_VERSION" ]; then 
            print_status "Python version is compatible"
        else
            print_error "Python 3.8 or higher is required"
            exit 1
        fi
    else
        print_error "Python 3 is not installed"
        exit 1
    fi
}

# Setup backend
setup_backend() {
    print_status "Setting up backend..."
    
    cd server
    
    # Create virtual environment if it doesn't exist
    if [ ! -d "venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv venv
    fi
    
    # Activate virtual environment
    print_status "Activating virtual environment..."
    source venv/bin/activate
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_status "Backend setup complete!"
    cd ..
}

# Create .env file
create_env_file() {
    print_status "Creating environment file..."
    
    if [ ! -f "server/.env" ]; then
        cp server/.env.example server/.env
        print_status "Created server/.env from example"
    else
        print_warning "server/.env already exists, skipping"
    fi
}

# Print next steps
print_next_steps() {
    echo ""
    echo "=========================================="
    echo "Setup Complete!"
    echo "=========================================="
    echo ""
    echo "Next steps:"
    echo ""
    echo "1. Start the backend server:"
    echo "   cd server"
    echo "   source venv/bin/activate"
    echo "   python main.py"
    echo ""
    echo "2. Install the Chrome extension:"
    echo "   - Open Chrome and go to chrome://extensions/"
    echo "   - Enable 'Developer mode' (toggle in top right)"
    echo "   - Click 'Load unpacked'"
    echo "   - Select the 'client' folder"
    echo ""
    echo "3. Test the API:"
    echo "   cd server"
    echo "   source venv/bin/activate"
    echo "   python test_api.py"
    echo ""
    echo "API Documentation: http://localhost:8000/docs"
    echo ""
}

# Main execution
main() {
    check_python
    setup_backend
    create_env_file
    print_next_steps
}

# Run main function
main