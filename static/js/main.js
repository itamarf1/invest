// Main JavaScript utilities for the Investment Dashboard

// Global utility functions
window.utils = {
    formatCurrency: (value) => {
        if (value === null || value === undefined) return 'N/A';
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'USD',
            minimumFractionDigits: 2,
            maximumFractionDigits: 2
        }).format(value);
    },
    
    formatNumber: (num) => {
        if (num === null || num === undefined) return 'N/A';
        if (num >= 1000000000) {
            return (num / 1000000000).toFixed(1) + 'B';
        } else if (num >= 1000000) {
            return (num / 1000000).toFixed(1) + 'M';
        } else if (num >= 1000) {
            return (num / 1000).toFixed(1) + 'K';
        }
        return num.toLocaleString();
    },
    
    formatPercent: (value, decimals = 2) => {
        if (value === null || value === undefined) return 'N/A';
        return value.toFixed(decimals) + '%';
    },
    
    formatDate: (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleDateString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric'
        });
    },
    
    formatDateTime: (dateString) => {
        if (!dateString) return 'N/A';
        const date = new Date(dateString);
        return date.toLocaleString('en-US', {
            year: 'numeric',
            month: 'short',
            day: 'numeric',
            hour: '2-digit',
            minute: '2-digit'
        });
    }
};

// Loading state management
window.loading = {
    show: () => {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.style.display = 'flex';
    },
    
    hide: () => {
        const overlay = document.getElementById('loadingOverlay');
        if (overlay) overlay.style.display = 'none';
    }
};

// Alert/notification system
window.alerts = {
    show: (message, type = 'error') => {
        const alertId = `${type}Alert`;
        const messageId = `${type}Message`;
        
        const alertElement = document.getElementById(alertId);
        const messageElement = document.getElementById(messageId);
        
        if (alertElement && messageElement) {
            messageElement.textContent = message;
            alertElement.style.display = 'block';
            
            // Auto-hide after 5 seconds for success messages
            if (type === 'success') {
                setTimeout(() => {
                    alertElement.style.display = 'none';
                }, 5000);
            }
        }
    },
    
    hide: (type = 'error') => {
        const alertElement = document.getElementById(`${type}Alert`);
        if (alertElement) alertElement.style.display = 'none';
    },
    
    error: (message) => window.alerts.show(message, 'error'),
    success: (message) => window.alerts.show(message, 'success'),
    warning: (message) => window.alerts.show(message, 'warning')
};

// API request wrapper with error handling
window.api = {
    request: async (url, data = null, method = 'POST') => {
        try {
            const config = {
                method: method,
                headers: {
                    'Content-Type': 'application/json'
                }
            };
            
            if (data && method !== 'GET') {
                config.data = data;
            }
            
            const response = await axios(url, config);
            return response.data;
        } catch (error) {
            console.error(`API request failed for ${url}:`, error);
            
            let errorMessage = 'An unexpected error occurred';
            
            if (error.response) {
                // Server responded with error status
                errorMessage = error.response.data?.detail || `Server error: ${error.response.status}`;
            } else if (error.request) {
                // Request made but no response
                errorMessage = 'Network error: Unable to reach server';
            }
            
            throw new Error(errorMessage);
        }
    }
};

// Chart color schemes
window.chartColors = {
    primary: '#2563eb',
    success: '#16a34a',
    danger: '#dc2626',
    warning: '#d97706',
    info: '#0891b2',
    gray: '#6b7280',
    
    // Color palettes for multiple data series
    palette: [
        '#2563eb', '#16a34a', '#dc2626', '#d97706', '#0891b2',
        '#7c3aed', '#db2777', '#059669', '#ea580c', '#0284c7'
    ],
    
    // Transparent versions for fills
    transparentPalette: [
        'rgba(37, 99, 235, 0.1)', 'rgba(22, 163, 74, 0.1)', 'rgba(220, 38, 38, 0.1)',
        'rgba(217, 119, 6, 0.1)', 'rgba(8, 145, 178, 0.1)', 'rgba(124, 58, 237, 0.1)',
        'rgba(219, 39, 119, 0.1)', 'rgba(5, 150, 105, 0.1)', 'rgba(234, 88, 12, 0.1)',
        'rgba(2, 132, 199, 0.1)'
    ]
};

// Common chart configuration
window.chartDefaults = {
    responsive: true,
    maintainAspectRatio: false,
    plugins: {
        legend: {
            display: true,
            position: 'top'
        }
    },
    scales: {
        x: {
            grid: {
                display: false
            }
        },
        y: {
            beginAtZero: false,
            grid: {
                color: '#f3f4f6'
            }
        }
    }
};

// Form validation helpers
window.validation = {
    isValidSymbol: (symbol) => {
        return /^[A-Z]{1,5}(-USD)?$/.test(symbol.toUpperCase());
    },
    
    isValidNumber: (value, min = null, max = null) => {
        const num = parseFloat(value);
        if (isNaN(num)) return false;
        if (min !== null && num < min) return false;
        if (max !== null && num > max) return false;
        return true;
    },
    
    cleanSymbol: (symbol) => {
        return symbol.trim().toUpperCase().replace(/[^A-Z-]/g, '');
    },
    
    parseSymbols: (input) => {
        return input.split(',')
                   .map(s => window.validation.cleanSymbol(s))
                   .filter(s => s.length > 0 && window.validation.isValidSymbol(s));
    }
};

// Local storage helpers
window.storage = {
    get: (key, defaultValue = null) => {
        try {
            const item = localStorage.getItem(key);
            return item ? JSON.parse(item) : defaultValue;
        } catch {
            return defaultValue;
        }
    },
    
    set: (key, value) => {
        try {
            localStorage.setItem(key, JSON.stringify(value));
            return true;
        } catch {
            return false;
        }
    },
    
    remove: (key) => {
        try {
            localStorage.removeItem(key);
            return true;
        } catch {
            return false;
        }
    }
};

// Initialize page-specific functionality
document.addEventListener('DOMContentLoaded', () => {
    // Set active navigation link
    const currentPath = window.location.pathname;
    const navLinks = document.querySelectorAll('.nav-links a');
    
    navLinks.forEach(link => {
        if (link.getAttribute('href') === currentPath) {
            link.classList.add('active');
        }
    });
    
    // Auto-hide alerts when clicking elsewhere
    document.addEventListener('click', (e) => {
        const alerts = document.querySelectorAll('.alert');
        alerts.forEach(alert => {
            if (!alert.contains(e.target) && alert.style.display === 'block') {
                // Don't auto-hide error alerts, only success/warning
                if (alert.classList.contains('alert-success') || alert.classList.contains('alert-warning')) {
                    alert.style.display = 'none';
                }
            }
        });
    });
});

// Global error handler
window.addEventListener('error', (event) => {
    console.error('Global error:', event.error);
    window.alerts.error('An unexpected error occurred. Please try again.');
});

// Handle unhandled promise rejections
window.addEventListener('unhandledrejection', (event) => {
    console.error('Unhandled promise rejection:', event.reason);
    window.alerts.error('An unexpected error occurred. Please try again.');
    event.preventDefault();
});