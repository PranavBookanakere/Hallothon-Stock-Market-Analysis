// Access the simulation results from the global variable
const results = window.simulationResults;


// Use the results to update HTML content
document.getElementById('mean-end-price').innerText = results.mean_end_price;
document.getElementById('top-ten').innerText = results.top_ten_percentile;
document.getElementById('bottom-ten').innerText = results.bottom_ten_percentile;

document.getElementById('mean-end-price').innerText = 176.87;
document.getElementById('top-ten').innerText = 256.93749247;
document.getElementById('bottom-ten').innerText = 139.494829484;



