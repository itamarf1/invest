# 📊 Algorithm Performance Dashboard

A comprehensive system for tracking, analyzing, and comparing the performance of all trading algorithms over time, even when they're not actively being used.

## 🚀 **Features Implemented**

### 🔍 **Performance Tracking System**
- **Multi-Algorithm Support**: Tracks Moving Average, RSI, MACD, Bollinger Bands, Sentiment Enhanced, and Ensemble strategies
- **Historical Analysis**: Continuously tracks performance over configurable time periods
- **Real-Time Monitoring**: Background analysis of algorithm performance across multiple symbols
- **Benchmark Comparison**: Compares against major market indices (SPY, QQQ, IWM, VTI, BND)

### 📈 **Comprehensive Metrics**
**Performance Metrics:**
- Total Return & Annualized Return
- Volatility (annualized)
- Sharpe Ratio
- Maximum Drawdown
- Win Rate & Profit Factor

**Risk Metrics:**
- Value at Risk (VaR 95%)
- Beta & Alpha vs benchmark
- Information Ratio
- Excess Return

**Trading Statistics:**
- Total/Winning/Losing Trades
- Average Win/Loss amounts
- Average Trade Duration
- Risk-adjusted returns

### 🎯 **Advanced Analytics**
- **Algorithm Comparison**: Side-by-side performance comparison
- **Performance Heat Maps**: Visual comparison across algorithms and symbols
- **Risk-Return Scatter Plots**: Risk-adjusted performance visualization
- **Benchmark Analysis**: Performance vs market indices
- **Top Performers Ranking**: Best algorithms by various metrics

### 🔧 **API Endpoints**

**Dashboard Overview** (`/api/algorithm-dashboard/`):
- `GET /overview` - Dashboard overview with key statistics
- `POST /run-analysis` - Run comprehensive analysis
- `GET /algorithms` - List all tracked algorithms
- `GET /algorithms/{algorithm_id}` - Detailed algorithm performance

**Performance Analysis**:
- `GET /algorithms/comparison` - Algorithm comparison by metrics
- `GET /performance-matrix` - Heat map data
- `GET /top-performers` - Top performing algorithms
- `GET /risk-analysis` - Risk analysis across algorithms

**Benchmarking**:
- `GET /benchmarks` - Benchmark performance data

**Admin Functions**:
- `GET /admin/system-status` - System health and statistics

### 🌐 **Dashboard UI** (`/algorithms`)

**Interactive Dashboard with:**
- **Overview Section**: Key metrics, top performers, algorithm type performance
- **Comparison Section**: Detailed side-by-side comparison table
- **Heat Map Section**: Visual performance matrix
- **Risk Analysis Section**: Risk-return scatter plots and drawdown analysis

**Features:**
- Real-time data updates
- Interactive charts (Chart.js)
- Responsive design (Bootstrap)
- Metric-based sorting and filtering
- Background analysis execution

## 🏗️ **Architecture**

### **Core Components**

1. **AlgorithmPerformanceTracker** (`src/analysis/algorithm_performance_tracker.py`)
   - Centralized performance tracking
   - Async analysis execution
   - Multi-symbol backtesting
   - Benchmark data management

2. **Dashboard Routes** (`src/analysis/algorithm_dashboard_routes.py`)
   - RESTful API endpoints
   - Authentication integration
   - Background task management
   - Admin functions

3. **Dashboard UI** (`templates/algorithm_dashboard.html`)
   - Interactive web interface
   - Real-time chart updates
   - Responsive design
   - User-friendly navigation

### **Data Flow**
1. **Data Collection**: Fetches historical market data for multiple symbols
2. **Signal Generation**: Runs all algorithms to generate trading signals
3. **Performance Simulation**: Simulates trading based on generated signals
4. **Metrics Calculation**: Computes comprehensive performance metrics
5. **Benchmark Comparison**: Compares against market indices
6. **Dashboard Display**: Presents data through interactive UI

### **Testing Symbols**
Default symbols analyzed: `["AAPL", "MSFT", "GOOGL", "TSLA", "SPY", "QQQ"]`

### **Benchmark Indices**
- **SPY**: S&P 500 (Primary benchmark)
- **QQQ**: NASDAQ 100
- **IWM**: Russell 2000
- **VTI**: Total Stock Market
- **BND**: Bond Market

## 🎯 **Usage Examples**

### **Starting Analysis**
```javascript
// Run comprehensive analysis
fetch('/api/algorithm-dashboard/run-analysis', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ lookback_days: 365 })
});
```

### **Getting Performance Comparison**
```javascript
// Get algorithm comparison sorted by Sharpe ratio
fetch('/api/algorithm-dashboard/algorithms/comparison?metric=sharpe_ratio&limit=20');
```

### **Risk Analysis**
```javascript
// Get comprehensive risk analysis
fetch('/api/algorithm-dashboard/risk-analysis');
```

## 📊 **Sample Output**

### **Performance Overview**
```json
{
  "total_algorithms_tracked": 36,
  "analysis_period": "2024-01-01 to 2024-11-01",
  "overall_statistics": {
    "average_return": 12.5,
    "median_return": 8.3,
    "best_return": 45.2,
    "worst_return": -15.7,
    "average_sharpe_ratio": 1.23,
    "average_max_drawdown": -8.4
  },
  "top_performers": {
    "best_return": {
      "name": "RSI Strategy (TSLA)",
      "return": 45.2,
      "sharpe_ratio": 2.1
    },
    "best_sharpe_ratio": {
      "name": "Ensemble Strategy (AAPL)",
      "return": 28.3,
      "sharpe_ratio": 2.8
    }
  }
}
```

### **Algorithm Comparison**
```json
{
  "comparison_metric": "total_return",
  "algorithms_compared": 36,
  "algorithms": [
    {
      "algorithm_id": "rsi_TSLA",
      "name": "RSI Strategy (TSLA)",
      "type": "rsi",
      "symbol": "TSLA",
      "total_return": 45.2,
      "annualized_return": 38.7,
      "sharpe_ratio": 2.1,
      "max_drawdown": -12.3,
      "win_rate": 68.5,
      "total_trades": 24,
      "excess_return": 31.8
    }
  ]
}
```

## 🔐 **Security & Authentication**

- **Protected Endpoints**: All API endpoints require authentication
- **Role-Based Access**: Admin endpoints require admin role
- **Rate Limiting**: Built-in protection against abuse
- **Session Management**: Secure session handling

## 🚀 **Getting Started**

### **Access the Dashboard**
1. Navigate to `http://localhost:8000/algorithms`
2. Login with Google OAuth (if not authenticated)
3. Click "Run Analysis" to generate fresh performance data
4. Explore different sections (Overview, Comparison, Heat Map, Risk)

### **Background Analysis**
The system automatically runs background analysis to keep performance data fresh. Analysis can also be triggered manually through the dashboard or API.

### **Navigation**
The Algorithm Performance Dashboard is integrated into the main navigation bar of the investment system.

## 📈 **Performance Insights**

**The dashboard provides insights into:**
- Which algorithms perform best on which symbols
- Risk-adjusted performance comparisons
- Market condition adaptation capabilities
- Trading frequency vs performance relationship
- Benchmark outperformance analysis

## 🔄 **Continuous Monitoring**

- **Auto-refresh**: Dashboard updates every 5 minutes
- **Background Analysis**: Periodic comprehensive analysis
- **Real-time Data**: Uses live market data for calculations
- **Historical Tracking**: Maintains performance history over time

## 🎯 **Next Steps**

The algorithm performance dashboard is now fully operational and provides:
1. ✅ Comprehensive algorithm tracking
2. ✅ Interactive performance dashboard
3. ✅ Real-time benchmark comparisons
4. ✅ Advanced risk analysis
5. ✅ Historical performance tracking

**This system enables data-driven algorithm selection and optimization for maximum investment returns!** 🚀