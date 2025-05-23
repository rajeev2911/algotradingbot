// Admin Panel JavaScript

// Global variables
let stocksData = [];
let forexData = [];
let optionsData = [];
let historyData = [];
let comexData = []; // New variable for COMEX data
let charts = {};

// Document ready function
$(document).ready(function() {
    // Initialize the admin panel
    initAdminPanel();
    
    // Set up event listeners
    setupEventListeners();
});

// Initialize the admin panel
function initAdminPanel() {
    // Load dashboard data
    loadDashboardData();
    
    // Load top stocks data
    loadTopStocks();
    
    // Load forex data
    loadTopForex();
    
    // Load COMEX commodities data
    loadTopComex();
    
    // Load options data
    loadKeyFutures();
    
    // Load historical data (default 7 days)
    loadHistoricalData(7);
}

// Set up event listeners
function setupEventListeners() {
    // Refresh buttons
    $('#refresh-data').click(function() {
        loadDashboardData();
    });
    
    $('#refresh-stocks').click(function() {
        loadTopStocks();
    });
    
    $('#refresh-forex').click(function() {
        loadTopForex();
    });
    
    $('#refresh-comex').click(function() {
        loadTopComex();
    });
    
    $('#refresh-options').click(function() {
        loadKeyFutures();
    });
    
    // History period buttons
    $('#history-7d').click(function() {
        loadHistoricalData(7);
    });
    
    $('#history-30d').click(function() {
        loadHistoricalData(30);
    });
    
    $('#history-90d').click(function() {
        loadHistoricalData(90);
    });
}

// Load dashboard data
function loadDashboardData() {
    // Show loading indicators
    $('#top-stock-name').text('Loading...');
    $('#top-forex-name').text('Loading...');
    $('#top-comex-name').text('Loading...');
    $('#market-trend').text('Loading...');
    
    // Load top stocks
    $.ajax({
        url: '/api/top-stocks',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            if (data && data.length > 0) {
                // Update top stock card
                $('#top-stock-name').text(data[0].name || data[0].ticker);
                $('#top-stock-price').text('$' + (data[0].current_price || 0).toFixed(2));
                
                // Update key stocks table
                updateKeyStocksTable(data);
                
                // Create stocks chart
                createStocksChart(data);
            }
        },
        error: function(xhr, status, error) {
            console.error('Error loading top stocks:', status, error);
            $('#top-stock-name').text('Error loading data');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
    
    // Load top forex
    $.ajax({
        url: '/api/top-forex',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            if (data && data.length > 0) {
                // Update top forex card
                $('#top-forex-name').text(data[0].name || data[0].ticker);
                $('#top-forex-price').text('$' + (data[0].current_price || 0).toFixed(4));
                
                // Create forex chart
                createForexChart(data);
            }
        },
        error: function(xhr, status, error) {
            console.error('Error loading top forex:', status, error);
            $('#top-forex-name').text('Error loading data');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
    
    // Load top COMEX commodities
    $.ajax({
        url: '/api/top-comex',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            if (data && data.length > 0) {
                // Update top commodity card
                $('#top-comex-name').text(data[0].name || data[0].ticker);
                $('#top-comex-price').text('$' + (data[0].current_price || 0).toFixed(2));
                
                // Create COMEX chart
                createComexChart(data);
            }
        },
        error: function(xhr, status, error) {
            console.error('Error loading top COMEX commodities:', status, error);
            $('#top-comex-name').text('Error loading data');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
    
    // Load key stocks
    $.ajax({
        url: '/api/key-stocks',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            if (data) {
                // Determine market trend based on top stocks
                let bullishCount = 0;
                let bearishCount = 0;
                
                if (data.top_overall) {
                    data.top_overall.forEach(stock => {
                        if (stock.rsi > 50) bullishCount++;
                        else bearishCount++;
                    });
                }
                
                let trend = bullishCount > bearishCount ? 'Bullish' : (bearishCount > bullishCount ? 'Bearish' : 'Neutral');
                $('#market-trend').text(trend);
                $('#market-trend-desc').text(`${bullishCount} bullish, ${bearishCount} bearish`);
            }
        },
        error: function(xhr, status, error) {
            console.error('Error loading key stocks:', status, error);
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
    
    // Load key futures/options
    $.ajax({
        url: '/api/key-futures',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            if (data && data.length > 0) {
                // Update key future card
                $('#key-future-name').text(data[0].symbol);
                $('#key-future-rec').text(data[0].recommendation);
            }
        },
        error: function(xhr, status, error) {
            console.error('Error loading key futures:', status, error);
            $('#key-future-name').text('Error loading data');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
}

// Load top stocks data
function loadTopStocks() {
    $.ajax({
        url: '/api/top-stocks',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            stocksData = data;
            updateTopStocksTable(data);
        },
        error: function(xhr, status, error) {
            console.error('Error loading top stocks:', status, error);
            $('#top-stocks-table').html('<tr><td colspan="10" class="text-center">Error loading data</td></tr>');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
}

// Update top stocks table
function updateTopStocksTable(data) {
    if (!data || data.length === 0) {
        $('#top-stocks-table').html('<tr><td colspan="10" class="text-center">No data available</td></tr>');
        return;
    }
    
    let html = '';
    data.forEach(stock => {
        html += `<tr>
            <td>${stock.ticker}</td>
            <td>${stock.name || ''}</td>
            <td>$${(stock.current_price || 0).toFixed(2)}</td>
            <td>${(stock.total_return * 100 || 0).toFixed(2)}%</td>
            <td>${(stock.recent_30d_return * 100 || 0).toFixed(2)}%</td>
            <td>${(stock.rsi || 0).toFixed(1)}</td>
            <td>${stock.above_sma50 ? '<span class="text-success">Yes</span>' : '<span class="text-danger">No</span>'}</td>
            <td>${stock.above_sma20 ? '<span class="text-success">Yes</span>' : '<span class="text-danger">No</span>'}</td>
            <td>${(stock.total_score || 0).toFixed(1)}</td>
            </tr>`;
            // <td><button class="btn btn-sm btn-primary view-stock-details" data-ticker="${stock.ticker}">View</button></td>
    });
    
    $('#top-stocks-table').html(html);
    
    // Add event listener for view details buttons
    $('.view-stock-details').click(function() {
        let ticker = $(this).data('ticker');
        loadStockDetails(ticker);
    });
}

// Update key stocks table on dashboard
function updateKeyStocksTable(data) {
    if (!data || data.length === 0) {
        $('#key-stocks-table').html('<tr><td colspan="8" class="text-center">No data available</td></tr>');
        return;
    }
    
    let html = '';
    data.slice(0, 5).forEach(stock => {
        let signal = '';
        if (stock.rsi > 70) signal = '<span class="text-danger">Overbought</span>';
        else if (stock.rsi < 30) signal = '<span class="text-success">Oversold</span>';
        else signal = '<span class="text-warning">Neutral</span>';
        
        html += `<tr>
            <td>${stock.ticker}</td>
            <td>${stock.name || ''}</td>
            <td>$${(stock.current_price || 0).toFixed(2)}</td>
            <td>${(stock.total_return * 100 || 0).toFixed(2)}%</td>
            <td>${(stock.rsi || 0).toFixed(1)}</td>
            <td>${(stock.macd || 0).toFixed(2)}</td>
            <td>${signal}</td>
            <td>${(stock.total_score || 0).toFixed(1)}</td>
        </tr>`;
    });
    
    $('#key-stocks-table').html(html);
}

// Load stock details
function loadStockDetails(ticker) {
    $.ajax({
        url: `/api/stock-details/${ticker.replace('.NS', '')}`,
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            displayStockDetails(data);
        },
        error: function(xhr, status, error) {
            console.error(`Error loading details for ${ticker}:`, status, error);
            alert(`Error loading details for ${ticker}`);
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
}

// Display stock details
function displayStockDetails(data) {
    // Show the details card
    $('#stock-details-card').show();
    
    // Update details
    $('#stock-details-name').text(data.name);
    $('#detail-price').text(`$${data.last_price.toFixed(2)}`);
    $('#detail-change').text(`${data.change_percent.toFixed(2)}%`);
    $('#detail-rsi').text(data.indicators.rsi.toFixed(2));
    $('#detail-macd').text(data.indicators.macd.toFixed(2));
    $('#detail-sma20').text(`$${data.indicators.sma_20.toFixed(2)}`);
    $('#detail-sma50').text(`$${data.indicators.sma_50.toFixed(2)}`);
    
    // Create chart
    createStockDetailChart(data);
}

// Load top forex data
function loadTopForex() {
    $.ajax({
        url: '/api/top-forex',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            forexData = data;
            updateTopForexTable(data);
            createForexPerformanceCharts(data);
        },
        error: function(xhr, status, error) {
            console.error('Error loading top forex:', status, error);
            $('#top-forex-table').html('<tr><td colspan="9" class="text-center">Error loading data</td></tr>');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
}

// Update top forex table
function updateTopForexTable(data) {
    if (!data || data.length === 0) {
        $('#top-forex-table').html('<tr><td colspan="9" class="text-center">No data available</td></tr>');
        return;
    }
    
    let html = '';
    data.forEach(forex => {
        let signal = '';
        if (forex.rsi > 70) signal = '<span class="text-danger">Overbought</span>';
        else if (forex.rsi < 30) signal = '<span class="text-success">Oversold</span>';
        else signal = '<span class="text-warning">Neutral</span>';
        
        html += `<tr>
            <td>${forex.ticker}</td>
            <td>${forex.name || ''}</td>
            <td>${(forex.current_price || 0).toFixed(4)}</td>
            <td>${(forex.total_return * 100 || 0).toFixed(2)}%</td>
            <td>${(forex.recent_30d_return * 100 || 0).toFixed(2)}%</td>
            <td>${(forex.rsi || 0).toFixed(1)}</td>
            <td>${(forex.momentum_10d * 100 || 0).toFixed(2)}%</td>
            <td>${(forex.total_score || 0).toFixed(1)}</td>
            <td>${signal}</td>
        </tr>`;
    });
    
    $('#top-forex-table').html(html);
}

// Load key futures/options data
function loadKeyFutures() {
    $.ajax({
        url: '/api/key-futures',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            optionsData = data;
            updateOptionsTable(data);
            createOptionsCharts(data);
        },
        error: function(xhr, status, error) {
            console.error('Error loading key futures:', status, error);
            $('#options-table').html('<tr><td colspan="10" class="text-center">Error loading data</td></tr>');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
}

// Update options table
function updateOptionsTable(data) {
    if (!data || data.length === 0) {
        $('#options-table').html('<tr><td colspan="10" class="text-center">No data available</td></tr>');
        return;
    }
    
    let html = '';
    data.forEach(option => {
        let recClass = 'text-warning';
        if (option.recommendation === 'Bullish') recClass = 'text-success';
        else if (option.recommendation === 'Bearish') recClass = 'text-danger';
        
        html += `<tr>
            <td>${option.symbol}</td>
            <td>${option.expiry}</td>
            <td>${option.strike}</td>
            <td>$${option.call_price.toFixed(2)}</td>
            <td>$${option.put_price.toFixed(2)}</td>
            <td>${option.call_oi.toLocaleString()}</td>
            <td>${option.put_oi.toLocaleString()}</td>
            <td>${option.call_iv.toFixed(1)}%</td>
            <td>${option.put_iv.toFixed(1)}%</td>
            <td class="${recClass}">${option.recommendation}</td>
        </tr>`;
    });
    
    $('#options-table').html(html);
}

// Load historical data
function loadHistoricalData(days) {
    $.ajax({
        url: `/api/history?days=${days}`,
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            historyData = data;
            updateHistoryDateSelector(data);
            createHistoryTrendChart(data);
        },
        error: function(xhr, status, error) {
            console.error('Error loading historical data:', status, error);
            $('#history-dates').html('Error loading historical data');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
}

// Update history date selector
function updateHistoryDateSelector(data) {
    if (!data || data.length === 0) {
        $('#history-dates').html('No historical data available');
        return;
    }
    
    let html = '';
    data.forEach((entry, index) => {
        html += `<button class="btn btn-outline-primary m-1 history-date-btn" data-index="${index}">${entry.date}</button>`;
    });
    
    $('#history-dates').html(html);
    
    // Add event listener for date buttons
    $('.history-date-btn').click(function() {
        let index = $(this).data('index');
        displayHistoricalData(index);
    });
    
    // Show first date by default
    displayHistoricalData(0);
}

// Display historical data for a specific date
function displayHistoricalData(index) {
    if (!historyData || !historyData[index]) return;
    
    const entry = historyData[index];
    
    // Update stocks table
    if (entry.top_stocks && entry.top_stocks.length > 0) {
        let html = '';
        entry.top_stocks.forEach(stock => {
            html += `<tr>
                <td>${stock.ticker}</td>
                <td>${stock.name || ''}</td>
                <td>$${(stock.current_price || 0).toFixed(2)}</td>
                <td>${(stock.total_return * 100 || 0).toFixed(2)}%</td>
                <td>${(stock.total_score || 0).toFixed(1)}</td>
            </tr>`;
        });
        $('#history-stocks-table').html(html);
    } else {
        $('#history-stocks-table').html('<tr><td colspan="5" class="text-center">No stock data available for this date</td></tr>');
    }
    
    // Update forex table
    if (entry.top_forex && entry.top_forex.length > 0) {
        let html = '';
        entry.top_forex.forEach(forex => {
            html += `<tr>
                <td>${forex.ticker}</td>
                <td>${forex.name || ''}</td>
                <td>${(forex.current_price || 0).toFixed(4)}</td>
                <td>${(forex.total_return * 100 || 0).toFixed(2)}%</td>
                <td>${(forex.total_score || 0).toFixed(1)}</td>
            </tr>`;
        });
        $('#history-forex-table').html(html);
    } else {
        $('#history-forex-table').html('<tr><td colspan="5" class="text-center">No forex data available for this date</td></tr>');
    }
}

// Load top COMEX commodities data
function loadTopComex() {
    $.ajax({
        url: '/api/top-comex',
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            comexData = data;
            updateTopComexTable(data);
            createComexPerformanceCharts(data);
        },
        error: function(xhr, status, error) {
            console.error('Error loading COMEX data:', status, error);
            $('#top-comex-table').html('<tr><td colspan="8" class="text-center">Error loading data</td></tr>');
            
            // Check if we need to reload the page due to server restart
            if (xhr.status === 0) {
                setTimeout(function() {
                    window.location.reload();
                }, 5000);
            }
        }
    });
}

// Update top COMEX commodities table
function updateTopComexTable(data) {
    // Clear existing table rows
    $('#top-comex-table').empty();
    
    // Add new rows
    $.each(data, function(i, commodity) {
        var row = '<tr>';
        row += '<td><a href="#" class="comex-details-link" data-ticker="' + commodity.ticker + '">' + commodity.ticker + '</a></td>';
        row += '<td>' + (commodity.name || commodity.ticker) + '</td>';
        row += '<td>$' + (commodity.current_price || 0).toFixed(2) + '</td>';
        row += '<td>' + (commodity.total_return || 0).toFixed(2) + '%</td>';
        row += '<td>' + (commodity.return_30d || 0).toFixed(2) + '%</td>';
        row += '<td>' + (commodity.rsi || 0).toFixed(1) + '</td>';
        row += '<td>' + (commodity.macd || 0).toFixed(2) + '</td>';
        row += '<td>' + (commodity.signal || 'NEUTRAL') + '</td>';
        row += '</tr>';
        
        $('#top-comex-table').append(row);
    });
    
    // Add click event for commodity details
    $('.comex-details-link').click(function(e) {
        e.preventDefault();
        var ticker = $(this).data('ticker');
        loadComexDetails(ticker);
    });
}

// Load COMEX commodity details
function loadComexDetails(ticker) {
    $.ajax({
        url: '/api/comex-details/' + ticker,
        method: 'GET',
        dataType: 'json',
        timeout: 30000, // 30 second timeout
        success: function(data) {
            displayComexDetails(data);
        },
        error: function(xhr, status, error) {
            console.error('Error loading COMEX details:', status, error);
            alert('Error loading details for ' + ticker);
        }
    });
}

// Display COMEX commodity details
function displayComexDetails(data) {
    // Update COMEX details card
    $('#comex-details-name').text(data.name);
    $('#comex-detail-price').text('$' + data.last_price.toFixed(2));
    $('#comex-detail-change').text(data.change_percent.toFixed(2) + '%');
    $('#comex-detail-rsi').text(data.indicators.rsi.toFixed(1));
    $('#comex-detail-macd').text(data.indicators.macd.toFixed(2));
    $('#comex-detail-sma20').text('$' + data.indicators.sma_20.toFixed(2));
    $('#comex-detail-sma50').text('$' + data.indicators.sma_50.toFixed(2));
    
    // Show the details card
    $('#comex-details-card').show();
    
    // Create commodity detail chart
    createComexDetailChart(data);
}

// Create stocks chart for dashboard
function createStocksChart(data) {
    if (!data || data.length === 0) return;
    
    // Prepare data
    const labels = data.slice(0, 5).map(stock => stock.name || stock.ticker);
    const returns = data.slice(0, 5).map(stock => stock.total_return * 100 || 0);
    
    // Destroy existing chart if it exists
    if (charts.stocksChart) {
        charts.stocksChart.destroy();
    }
    
    // Create new chart
    const ctx = document.getElementById('stocksChart').getContext('2d');
    charts.stocksChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Return (%)',
                data: returns,
                backgroundColor: 'rgba(54, 162, 235, 0.5)',
                borderColor: 'rgba(54, 162, 235, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Top 5 Stocks by Return'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Return (%)'
                    }
                }
            }
        }
    });
}

// Create forex chart for dashboard
function createForexChart(data) {
    if (!data || data.length === 0) return;
    
    // Prepare data
    const labels = data.slice(0, 5).map(forex => forex.name || forex.ticker);
    const returns = data.slice(0, 5).map(forex => forex.total_return * 100 || 0);
    
    // Destroy existing chart if it exists
    if (charts.forexChart) {
        charts.forexChart.destroy();
    }
    
    // Create new chart
    const ctx = document.getElementById('forexChart').getContext('2d');
    charts.forexChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Return (%)',
                data: returns,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Top 5 Forex Pairs by Return'
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    title: {
                        display: true,
                        text: 'Return (%)'
                    }
                }
            }
        }
    });
}

// Create stock detail chart
function createStockDetailChart(data) {
    if (!data || !data.data || data.data.length === 0) return;
    
    // Prepare data
    const prices = data.data.map(d => d.Close);
    const dates = data.data.map(d => new Date(d.Date).toLocaleDateString());
    const sma20 = data.data.map(d => d.sma_short);
    const sma50 = data.data.map(d => d.sma_long);
    
    // Destroy existing chart if it exists
    if (charts.stockDetailChart) {
        charts.stockDetailChart.destroy();
    }
    
    // Create new chart
    const ctx = document.getElementById('stockDetailChart').getContext('2d');
    charts.stockDetailChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Price',
                    data: prices,
                    borderColor: 'rgba(54, 162, 235, 1)',
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    fill: true,
                    tension: 0.1
                },
                {
                    label: 'SMA 20',
                    data: sma20,
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                },
                {
                    label: 'SMA 50',
                    data: sma50,
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    pointRadius: 0,
                    fill: false
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: `${data.name} Price History`
                }
            },
            scales: {
                x: {
                    ticks: {
                        maxTicksLimit: 10
                    }
                }
            }
        }
    });
}

// Create forex performance charts
function createForexPerformanceCharts(data) {
    if (!data || data.length === 0) return;
    
    // Prepare data for performance chart
    const labels = data.slice(0, 5).map(forex => forex.name || forex.ticker);
    const returns = data.slice(0, 5).map(forex => forex.total_return * 100 || 0);
    
    // Destroy existing chart if it exists
    if (charts.forexPerformanceChart) {
        charts.forexPerformanceChart.destroy();
    }
    
    // Create performance chart
    const perfCtx = document.getElementById('forexPerformanceChart').getContext('2d');
    charts.forexPerformanceChart = new Chart(perfCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Total Return (%)',
                data: returns,
                backgroundColor: 'rgba(75, 192, 192, 0.5)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Forex Performance'
                }
            }
        }
    });
    
    // Prepare data for volatility chart
    const volatility = data.slice(0, 5).map(forex => forex.volatility * 100 || 0);
    
    // Destroy existing chart if it exists
    if (charts.forexVolatilityChart) {
        charts.forexVolatilityChart.destroy();
    }
    
    // Create volatility chart
    const volCtx = document.getElementById('forexVolatilityChart').getContext('2d');
    charts.forexVolatilityChart = new Chart(volCtx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Volatility (%)',
                data: volatility,
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1
            }]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Forex Volatility'
                }
            }
        }
    });
}

// Create options charts
function createOptionsCharts(data) {
    if (!data || data.length === 0) return;
    
    // Prepare data for OI chart
    const labels = data.map(option => option.symbol);
    const callOI = data.map(option => option.call_oi);
    const putOI = data.map(option => option.put_oi);
    
    // Destroy existing chart if it exists
    if (charts.optionsOIChart) {
        charts.optionsOIChart.destroy();
    }
    
    // Create OI chart
    const oiCtx = document.getElementById('optionsOIChart').getContext('2d');
    charts.optionsOIChart = new Chart(oiCtx, {
        type: 'bar',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Call OI',
                    data: callOI,
                    backgroundColor: 'rgba(54, 162, 235, 0.5)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 1
                },
                {
                    label: 'Put OI',
                    data: putOI,
                    backgroundColor: 'rgba(255, 99, 132, 0.5)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Options Open Interest'
                }
            },
            scales: {
                y: {
                    beginAtZero: true
                }
            }
        }
    });
    
    // Prepare data for IV chart
    const callIV = data.map(option => option.call_iv);
    const putIV = data.map(option => option.put_iv);
    
    // Destroy existing chart if it exists
    if (charts.optionsIVChart) {
        charts.optionsIVChart.destroy();
    }
    
    // Create IV chart
    const ivCtx = document.getElementById('optionsIVChart').getContext('2d');
    charts.optionsIVChart = new Chart(ivCtx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Call IV (%)',
                    data: callIV,
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Put IV (%)',
                    data: putIV,
                    backgroundColor: 'rgba(255, 99, 132, 0.1)',
                    borderColor: 'rgba(255, 99, 132, 1)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Options Implied Volatility'
                }
            }
        }
    });
}

// Create history trend chart
function createHistoryTrendChart(data) {
    if (!data || data.length === 0) return;
    
    // Prepare data
    const dates = data.map(entry => entry.date);
    
    // Calculate average scores for stocks and forex
    const stockScores = data.map(entry => {
        if (!entry.top_stocks || entry.top_stocks.length === 0) return 0;
        const scores = entry.top_stocks.map(stock => stock.total_score || 0);
        return scores.reduce((a, b) => a + b, 0) / scores.length;
    });
    
    const forexScores = data.map(entry => {
        if (!entry.top_forex || entry.top_forex.length === 0) return 0;
        const scores = entry.top_forex.map(forex => forex.total_score || 0);
        return scores.reduce((a, b) => a + b, 0) / scores.length;
    });
    
    // Destroy existing chart if it exists
    if (charts.historyTrendChart) {
        charts.historyTrendChart.destroy();
    }
    
    // Create trend chart
    const ctx = document.getElementById('historyTrendChart').getContext('2d');
    charts.historyTrendChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Avg Stock Score',
                    data: stockScores,
                    backgroundColor: 'rgba(54, 162, 235, 0.1)',
                    borderColor: 'rgba(54, 162, 235, 1)',
                    borderWidth: 2,
                    tension: 0.1
                },
                {
                    label: 'Avg Forex Score',
                    data: forexScores,
                    backgroundColor: 'rgba(75, 192, 192, 0.1)',
                    borderColor: 'rgba(75, 192, 192, 1)',
                    borderWidth: 2,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            plugins: {
                title: {
                    display: true,
                    text: 'Performance Score Trends'
                }
            }
        }
    });
}

// Create COMEX chart for dashboard
function createComexChart(data) {
    // Prepare data for chart
    var labels = [];
    var prices = [];
    var returns = [];
    
    $.each(data, function(i, commodity) {
        if (i < 5) {  // Only show top 5
            labels.push(commodity.name || commodity.ticker);
            prices.push(commodity.current_price || 0);
            returns.push(commodity.total_return || 0);
        }
    });
    
    // Create chart
    var ctx = document.getElementById('comex-performance-chart');
    if (ctx) {
        if (charts.comexPerformance) {
            charts.comexPerformance.destroy();
        }
        
        charts.comexPerformance = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Return (%)',
                    data: returns,
                    backgroundColor: 'rgba(255, 193, 7, 0.5)',
                    borderColor: 'rgba(255, 193, 7, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Return (%)'
                        }
                    }
                }
            }
        });
    }
}

// Create COMEX performance charts
function createComexPerformanceCharts(data) {
    // Prepare data for performance chart
    var labels = [];
    var returns = [];
    var volatility = [];
    
    $.each(data, function(i, commodity) {
        if (i < 5) {  // Only show top 5
            labels.push(commodity.name || commodity.ticker);
            returns.push(commodity.total_return || 0);
            volatility.push(commodity.volatility || 0);
        }
    });
    
    // Create performance chart
    var perfCtx = document.getElementById('comex-performance-chart');
    if (perfCtx) {
        if (charts.comexPerformance) {
            charts.comexPerformance.destroy();
        }
        
        charts.comexPerformance = new Chart(perfCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Return (%)',
                    data: returns,
                    backgroundColor: 'rgba(255, 193, 7, 0.5)',
                    borderColor: 'rgba(255, 193, 7, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Return (%)'
                        }
                    }
                }
            }
        });
    }
    
    // Create volatility chart
    var volCtx = document.getElementById('comex-volatility-chart');
    if (volCtx) {
        if (charts.comexVolatility) {
            charts.comexVolatility.destroy();
        }
        
        charts.comexVolatility = new Chart(volCtx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Volatility (%)',
                    data: volatility,
                    backgroundColor: 'rgba(220, 53, 69, 0.5)',
                    borderColor: 'rgba(220, 53, 69, 1)',
                    borderWidth: 1
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    y: {
                        beginAtZero: true,
                        title: {
                            display: true,
                            text: 'Volatility (%)'
                        }
                    }
                }
            }
        });
    }
}

// Create COMEX detail chart
function createComexDetailChart(data) {
    var ctx = document.getElementById('comex-detail-chart');
    if (!ctx) return;
    
    if (charts.comexDetail) {
        charts.comexDetail.destroy();
    }
    
    // Prepare data
    var prices = [];
    var dates = [];
    var sma20 = [];
    var sma50 = [];
    
    if (data.data && data.data.length > 0) {
        $.each(data.data, function(i, point) {
            if (i % 3 === 0) {  // Sample every 3rd point to avoid overcrowding
                dates.push(new Date(point.Date).toLocaleDateString());
                prices.push(point.Close);
                sma20.push(point.sma_short);
                sma50.push(point.sma_long);
            }
        });
    }
    
    // Create chart
    charts.comexDetail = new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [
                {
                    label: 'Price',
                    data: prices,
                    borderColor: 'rgba(255, 193, 7, 1)',
                    backgroundColor: 'rgba(255, 193, 7, 0.1)',
                    borderWidth: 2,
                    fill: false
                },
                {
                    label: 'SMA 20',
                    data: sma20,
                    borderColor: 'rgba(40, 167, 69, 1)',
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0
                },
                {
                    label: 'SMA 50',
                    data: sma50,
                    borderColor: 'rgba(0, 123, 255, 1)',
                    borderWidth: 1,
                    fill: false,
                    pointRadius: 0
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Date'
                    }
                },
                y: {
                    display: true,
                    title: {
                        display: true,
                        text: 'Price ($)'
                    }
                }
            }
        }
    });
}