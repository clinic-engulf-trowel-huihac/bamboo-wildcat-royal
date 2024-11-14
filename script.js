// Array of CSV file paths
const csvFilePaths = [
    'data/model_accuracy.csv',
    'data/A100_PT.csv',
    'data/A100_TRT.csv',
    'data/H100_PT.csv',
    'data/H100_TRT.csv'
];

const accuracyColumns = ['imagenet-original', 'imagenet-real', 'imagenetv2-matched-frequency', 'sketch', 'imagenet-a', 'imagenet-a-clean', 'imagenet-r', 'imagenet-r-clean'];
// Populate the accuracy selectors
const selectorDiv = document.getElementById('accuracy-selectors');
accuracyColumns.forEach(function(acc, index) {
    const label = document.createElement('label');
    // Only check the first checkbox by default
    const checkedAttribute = index === 0 ? 'checked' : '';
    label.innerHTML = `<input type="checkbox" value="${acc}" ${checkedAttribute}> ${acc}`;
    selectorDiv.appendChild(label);
});

// Function to compute 'Score' based on 'Weight' using vectorized operations
function computeScore(Weight, x, y, xMax) {
    const xLength = x.length;
    const yLength = y.length;
    const score = [];

    if (document.querySelector('input[name="score-select"]:checked').value === 'manhattan') {
        for (let i = 0; i < yLength; i++) {
            const row = new Float32Array(xLength); // Use typed array for row
            for (let j = 0; j < xLength; j++) row[j] = Weight * (1 - x[j]/xMax) + y[i] * (1 - Weight/100);
            score.push(row);
        }
    } else if (document.querySelector('input[name="score-select"]:checked').value === 'accuracyperjoule') {
        for (let i = 0; i < yLength; i++) {
            const row = new Float32Array(xLength); // Use typed array for row
            for (let j = 0; j < xLength; j++)  row[j] = y[i] / (x[j] + 0.01);
            score.push(row);
        }
    }

    return score;
}

function computeSingleScore(Weight, x, y, xMax) {
    // return score = Math.sqrt(Weight / 1e5 * x * x + (y - 100) * (y - 100));
    if (document.querySelector('input[name="score-select"]:checked').value === 'manhattan') {
        return Weight * (1 - x/xMax) + y * (1 - Weight/100);
    } else if (document.querySelector('input[name="score-select"]:checked').value === 'accuracyperjoule') {
        return y/x;
    }
}

function calcXRanges(xMin, xMax) {
    let xFloor = Math.pow(10, Math.floor(Math.log10(xMin)));
    let xCeiling = Math.pow(10, Math.ceil(Math.log10(xMax)));

    if (xFloor * 5 < xMin) xFloor *= 5;      else if (xFloor * 2 < xMin) xFloor *= 2;
    if (xCeiling / 5 > xMax) xCeiling /= 5;  else if (xCeiling / 2 > xMax) xCeiling /= 2;

    return [xFloor, xCeiling];
}


// Load CSV data using D3.js
Promise.all(csvFilePaths.map(file => d3.csv(file))).then(function(datasets) {
    const [accuracy, A100_PT, A100_TRT, H100_PT, H100_TRT] = datasets;

    // Add event listeners to checkboxes
    document.querySelectorAll('#accuracy-selectors input').forEach(function(checkbox) {checkbox.addEventListener('change', updateAccuracy)});

    // Add event listener to GPU selector
    document.querySelectorAll('#gpu-selectors input[name="gpu-select"]').forEach(radio => {radio.addEventListener('change', updateEnergy)});
    document.querySelectorAll('#deployment-selectors input[name="deployment-select"]').forEach(radio => {radio.addEventListener('change', updateEnergy)});

    // Add event listener to xscale selector
    document.querySelectorAll('#xscale-selectors input[name="xscale-select"]').forEach(radio => {
        radio.addEventListener('change', function() {
            Plotly.relayout('plotDiv', {'xaxis.type': this.value});
        });
    });

    // Add event listener to score selector
    document.querySelectorAll('#score-selectors input[name="score-select"]').forEach(radio => {radio.addEventListener('change', updateScoreConfig)});

    document.getElementById('weightSlider').addEventListener('input', function() {
        Weight = parseFloat(this.value);
        document.getElementById('weightValue').textContent = Weight;
        updateScoreConfig();
    });

    document.getElementById('min-accuracy').addEventListener('input', updateAcceptableRegion);
    document.getElementById('max-energy').addEventListener('input', updateAcceptableRegion);
    document.getElementById('top-scores').addEventListener('input', updateScoreBoard);

    // Prepare custom data for click events
    const customData = accuracy.map(d => d.url);

    let xData, yData;
    let model_name, input_size, throughput, latency;
    let scatter, contour, horizontalLine, verticalLine, layout, config;
    let xMin, xMax, yMin, yMax, xFloor, xCeiling;
    let scoreBoard = [];
    const xPoints = 1000;
    const yPoints = 100;
    let xGrid, yGrid;
    const contourStart = 20;
    const contourDensity = 5;

    firstPlot();

    function firstPlot() {
        const selectedAccuracies = Array.from(document.querySelectorAll('#accuracy-selectors input:checked')).map(input => input.value);
        const selectedAccuraciesLength = selectedAccuracies.length;

        // Prepare data for plotting
        let data;

        // Get the selected values for GPU and deployment
        const gpu = document.querySelector('#gpu-selectors input:checked').value;
        const deployment = document.querySelector('#deployment-selectors input:checked').value;
        
        // Select the appropriate dataset based on the chosen GPU and deployment type
        if (gpu === 'A100') data = deployment === 'PT' ? A100_PT : A100_TRT;
        else if (gpu === 'H100') data = deployment === 'PT' ? H100_PT : H100_TRT;

        // const xData = data.map(d => parseFloat(d['Energy']));
        xData = Float32Array.from(data, d => parseFloat(d['Energy']));

        model_name = data.map(d => d['Model Name']);
        input_size = data.map(d => d['Input Size']);
        throughput = data.map(d => parseFloat(d['Throughput']));
        latency = data.map(d => parseFloat(d['Latency']));

        yData = Float32Array.from(accuracy, d => {
            const sum = selectedAccuracies.reduce((acc, curr) => acc + parseFloat(d[curr] || 0), 0);
            return sum / selectedAccuraciesLength;
        });
        
        // Prepare hover text
        const hoverText = data.map((d, i) => 
            ` <b>${model_name[i]}</b> <br>` +
            ` Input Size:    ${input_size[i]} <br>` +
            ` Accuracy:      ${yData[i].toFixed(3)} % <br>` +
            ` Energy:        ${xData[i].toFixed(3)} mJ <br>` +
            ` Throughput:    ${throughput[i].toFixed(3)} img/s <br>` +
            ` Latency:       ${latency[i].toFixed(3)} ms <br>`
        );

        scatter = {
            x: xData, y: yData,
            mode: 'markers',
            type: 'scatter',
            text: hoverText,
            hoverinfo: 'text',
            customdata: customData,
            marker: {size: 4, color: 'black'},
        };

        // Use reduce to find min and max values
        xMin = xData.reduce((min, val) => (val < min ? val : min), Infinity);
        xMax = xData.reduce((max, val) => (val > max ? val : max), -Infinity);
    
        [xFloor, xCeiling] = calcXRanges(xMin, xMax);

        yMin = Math.round(yData.reduce((min, val) => (val < min ? val : min), Infinity));
        yMax = Math.round(yData.reduce((max, val) => (val > max ? val : max), -Infinity));

        // Generate x and y arrays using typed arrays for better performance
        const weight = parseFloat(document.getElementById('weightSlider').value);
        yGrid = Float32Array.from({ length: yPoints }, (_, i) => i * (100 / (yPoints - 1)));
        xGrid = Float32Array.from({ length: xPoints }, (_, i) => Math.pow(10, Math.log10(xFloor) + i * (Math.log10(xCeiling / xFloor) / (xPoints - 1))));

        // Compute the score grid
        const score = computeScore(weight, xGrid, yGrid, xMax);

        // Create initial contour plot
        contour = {
            x: xGrid, y: yGrid, z: score,
            type: 'contour',
            contours: {start: contourStart, end: 100, size: contourDensity},
            line: {smoothing: 0},
            showscale: true,
            reversescale: true,
            opacity: 0.32,
            hoverinfo: 'skip',
            autocontour: false,
        };

        document.getElementById('min-accuracy').value = yMin-1;
        document.getElementById('max-energy').value = Math.ceil(xMax);
        horizontalLine = {
            x: [xFloor, Math.ceil(xMax)], // Start and end x-values (use your data range or axis range)
            y: [yMin-1, yMin-1], // Same y-value for a horizontal line
            mode: 'lines',
            line: {
                color: 'red',
                width: 1,
                dash: 'solid', // Options: 'solid', 'dot', 'dash', etc.
            },
            hoverinfo: 'skip',   // Optional: Exclude from hover events
            showlegend: false,   // Optional: Exclude from the legend
        };

        verticalLine = {
            x: [Math.ceil(xMax), Math.ceil(xMax)], // Same x-value for a vertical line
            y: [100, document.getElementById('min-accuracy').value], // Start and end y-values (use your data range or axis range)
            mode: 'lines',
            line: {
                color: 'red',
                width: 1,
                dash: 'solid', // Options: 'solid', 'dot', 'dash', etc.
            },
            hoverinfo: 'skip',   // Optional: Exclude from hover events
            showlegend: false,   // Optional: Exclude from the legend
        };

        // Define the layout
        layout = {
            margin: { t: 24, r: 24, b: 64, l: 64 },
            showlegend: false,
            hovermode: 'closest',
            hoverlabel: { 
                align: 'left',
                bgcolor: 'white',
                font: {family: 'Courier New', size: 14}
                
            },
            xaxis: {
                title: 'Energy (mJ)',
                type: document.querySelector('#xscale-selectors input:checked').value,
                range: [Math.log10(xFloor), Math.log10(xCeiling)],
                showspikes: true,
                spikemode: 'across',
                spikethickness: 1,
                spikesnap: 'data',
                showline: true,
            },
            yaxis: {
                title: 'Accuracy (%)',
                range: [yMin-4, yMax+4],
                showspikes: true,
                spikemode: 'across',
                spikethickness: 1,
                spikesnap: 'data',
                showline: true
            }
        };

        config = {responsive: true};
        
        // Render the plot
        Plotly.react('plotDiv', [contour, horizontalLine, verticalLine, scatter], layout, config);

        updateScoreBoard();
    }

    function updateAccuracy() {
        const selectedAccuracies = Array.from(document.querySelectorAll('#accuracy-selectors input:checked')).map(input => input.value);
        const selectedAccuraciesLength = selectedAccuracies.length;

        // Ensure at least one accuracy is selected
        if (selectedAccuraciesLength === 0) return;
        
        yData = Float32Array.from(accuracy, d => {
            const sum = selectedAccuracies.reduce((acc, curr) => acc + parseFloat(d[curr] || 0), 0);
            return sum / selectedAccuraciesLength;
        });

        // Prepare hover text
        const hoverText = model_name.map((d, i) => 
            ` <b>${model_name[i]}</b> <br>` +
            ` Input Size:    ${input_size[i]} <br>` +
            ` Accuracy:      ${yData[i].toFixed(3)} % <br>` +
            ` Energy:        ${xData[i].toFixed(3)} mJ <br>` +
            ` Throughput:    ${throughput[i].toFixed(3)} img/s <br>` +
            ` Latency:       ${latency[i].toFixed(3)} ms <br>`
        );
        
        yMin = yData.reduce((min, val) => (val < min ? val : min), Infinity);
        yMax = yData.reduce((max, val) => (val > max ? val : max), -Infinity);

        scatter.y = yData;
        scatter.text = hoverText;
        layout.yaxis.range = [yMin-4, yMax+4];

        // Update the contour
        if (document.querySelector('input[name="score-select"]:checked').value === 'manhattan') {
            weight = parseFloat(document.getElementById('weightSlider').value);
            contour.z = computeScore(weight, xGrid, yGrid, xMax);
            contour.contours.start = contourStart;
            contour.contours.end = 100;
            contour.contours.size = contourDensity;

        } else if (document.querySelector('input[name="score-select"]:checked').value === 'accuracyperjoule') {
            contour.z = computeScore(0, xGrid, yGrid, xMax);
            contour.contours.start = yMax/xMax;
            contour.contours.end = yMin/xMin;
            contour.contours.size = (yMax/xMax - yMin/xMin) / 10;
        }


        Plotly.react('plotDiv', [contour, horizontalLine, verticalLine, scatter], layout, config);
        updateScoreBoard();
    }

    function updateEnergy() {
        let data;
        const gpu = document.querySelector('#gpu-selectors input:checked').value;
        const deployment = document.querySelector('#deployment-selectors input:checked').value;
        
        if (gpu === 'A100') data = deployment === 'PT' ? A100_PT : A100_TRT;
        else if (gpu === 'H100') data = deployment === 'PT' ? H100_PT : H100_TRT;

        xData = Float32Array.from(data, d => parseFloat(d['Energy']));

        throughput = data.map(d => parseFloat(d['Throughput']));
        latency = data.map(d => parseFloat(d['Latency']));

        const hoverText = model_name.map((d, i) => 
            ` <b>${model_name[i]}</b> <br>` +
            ` Input Size:    ${input_size[i]} <br>` +
            ` Accuracy:      ${yData[i].toFixed(3)} % <br>` +
            ` Energy:        ${xData[i].toFixed(3)} mJ <br>` +
            ` Throughput:    ${throughput[i].toFixed(3)} img/s <br>` +
            ` Latency:       ${latency[i].toFixed(3)} ms <br>`
        );

        scatter.x = xData;
        scatter.text = hoverText;

        xMin = xData.reduce((min, val) => (val < min ? val : min), Infinity);
        xMax = xData.reduce((max, val) => (val > max ? val : max), -Infinity);
    
        [xFloor, xCeiling] = calcXRanges(xMin, xMax);

        horizontalLine.x = [xFloor, Math.ceil(xMax)];
        verticalLine.x = [Math.ceil(xMax), Math.ceil(xMax)];
        document.getElementById('max-energy').value = Math.ceil(xMax);
        if (document.querySelector('input[name="xscale-select"]:checked').value === 'log') {
            layout.xaxis.range = [Math.log10(xFloor), Math.log10(xCeiling)];
        } else if (document.querySelector('input[name="xscale-select"]:checked').value === 'linear') {
            layout.xaxis.range = [xFloor, xCeiling];
        }

        // Update the contour
        xGrid = Float32Array.from({ length: xPoints }, (_, i) => Math.pow(10, Math.log10(xFloor) + i * (Math.log10(xCeiling / xFloor) / (xPoints - 1))));
        contour.x = xGrid;
        if (document.querySelector('input[name="score-select"]:checked').value === 'manhattan') {
            weight = parseFloat(document.getElementById('weightSlider').value);
            contour.z = computeScore(weight, xGrid, yGrid, xMax);
            contour.contours.start = contourStart;
            contour.contours.end = 100;
            contour.contours.size = contourDensity;

        } else if (document.querySelector('input[name="score-select"]:checked').value === 'accuracyperjoule') {
            contour.z = computeScore(0, xGrid, yGrid, xMax);
            contour.contours.start = yMax/xMax;
            contour.contours.end = yMin/xMin;
            contour.contours.size = (yMax/xMax - yMin/xMin) / 10;
        }
        
        // Render the plot
        Plotly.react('plotDiv', [contour, horizontalLine, verticalLine, scatter], layout, config);
        updateScoreBoard();
    }

    function updateAcceptableRegion() {
        const minAccuracy = parseFloat(document.getElementById('min-accuracy').value);
        const maxEnergy = parseFloat(document.getElementById('max-energy').value);

        horizontalLine.x = [xFloor, maxEnergy];
        verticalLine.x = [maxEnergy, maxEnergy];

        horizontalLine.y = [minAccuracy, minAccuracy];
        verticalLine.y = [100, minAccuracy];

        updateScoreBoard();
        Plotly.react('plotDiv', [contour, horizontalLine, verticalLine, scatter], layout, config);
    }

    function updateScoreConfig() {
        const weight = parseFloat(document.getElementById('weightSlider').value);
        const score = computeScore(weight, xGrid, yGrid, xMax);
        let updateOptions;

        if (document.querySelector('input[name="score-select"]:checked').value === 'manhattan') {
            updateOptions = {
                z: [score],
                contours: {start: contourStart, end: 100, size: contourDensity},
            };

        } else if (document.querySelector('input[name="score-select"]:checked').value === 'accuracyperjoule') {
            updateOptions = {
                z: [score],
                contours: {start: yMax/xMax, end: yMin/xMin, size: (yMax/xMax - yMin/xMin) / 10},
            };
        }

        Plotly.restyle('plotDiv', updateOptions, 0);

        updateScoreBoard();
    }
    
    function updateScoreBoard() {
        const x_threshold = parseFloat(document.getElementById('max-energy').value);
        const y_threshold = parseFloat(document.getElementById('min-accuracy').value);
        const weight = parseFloat(document.getElementById('weightSlider').value);

        // empty the scoreBoard
        scoreBoard = [];

        for (var i = 0; i < xData.length; i++) {
            if (xData[i] <= x_threshold && yData[i] >= y_threshold) {
                scoreBoard.push({
                    'score': computeSingleScore(weight, xData[i], yData[i], xMax),
                    'Model Name': accuracy[i]['Model Name'],
                    'Input Size': accuracy[i]['Input Size'],
                    'Energy': xData[i],
                    'Accuracy': yData[i],
                    'url': accuracy[i]['url'],
                });
            }
        }

        scoreBoard.sort((a, b) => b.score - a.score);
        // select the top 10 models
        scoreBoard = scoreBoard.slice(0, parseInt(document.getElementById('top-scores').value));
        // reverse the order
        scoreBoard.reverse();
        // find the max and min accuracy  
        const maxAccuracy = scoreBoard.reduce((max, val) => (val['Accuracy'] > max ? val['Accuracy'] : max), -Infinity);
        const minAccuracy = scoreBoard.reduce((min, val) => (val['Accuracy'] < min ? val['Accuracy'] : min), Infinity);

        const maxEnergy = scoreBoard.reduce((max, val) => (val['Energy'] > max ? val['Energy'] : max), -Infinity);
        const minEnergy = scoreBoard.reduce((min, val) => (val['Energy'] < min ? val['Energy'] : min), Infinity);

        let top_models = scoreBoard.map(d => d['Model Name'] + '.' + d['Input Size']);

        const bar_1_template = scoreBoard.map(d => d['Energy'].toFixed(2) + ' mJ');
        var bar_1 = {
            x: scoreBoard.map(d => d['Energy']),
            y: top_models,
            text: bar_1_template,
            textposition: 'outside',
            hoverinfo: 'none',
            name: 'Energy',
            type: 'bar',
            orientation: 'h',
            offsetgroup: 1,
        }

        const bar_2_template = scoreBoard.map(d => d['Accuracy'].toFixed(2) + '%');
        var bar_2 = {
            x: scoreBoard.map(d => d['Accuracy']),
            y: top_models,
            text: bar_2_template,
            textposition: 'outside',
            hoverinfo: 'none',
            name: 'Accuracy',
            type: 'bar',
            orientation: 'h',
            xaxis: 'x2',
            offsetgroup: 2,
        }

        let truncated_labels = top_models.map(d => d.slice(0, 20) + '...');

        var layout = {
            margin: { t: 32, r: 16, b: 48, l: 168 },
            barmode: 'group',
            xaxis: {
                type: 'log',
                title: 'Energy (mJ)',
                range: [Math.log10(minEnergy)-0.08, Math.log10(maxEnergy)+0.32],
                automargin: "true",
            },
            xaxis2: {
                title: 'Accuracy (%)',
                overlaying: 'x',
                side: 'top',
                range: [minAccuracy - 2, maxAccuracy + Math.max((maxAccuracy - minAccuracy) * 0.32, 2)],
                automargin: 'true',
            },
            yaxis: {
                showspikes: true,
                spikecolor: 'black',
                spikemode: 'across',
                spikedash: 'solid',
                spikethickness: 4,
                tickvals: top_models,
                ticktext: truncated_labels,
            },
            showlegend: false,

        }

        Plotly.react('scoreDiv', [bar_1, bar_2], layout);
    }
    
    // Add click event listener to the plot
    const plotDiv = document.getElementById('plotDiv');
    plotDiv.on('plotly_click', function(eventData) {
        const point = eventData.points[0];
        const url = point.customdata;
        if (url) window.open(url, '_blank');
        
    });

    // Add click event listener to the plot
    const scoreDiv = document.getElementById('scoreDiv');
    scoreDiv.on('plotly_click', function(eventData) {
        const point = eventData.points[0];
        const url = scoreBoard[point.pointNumber]['url'];
        if (url) window.open(url, '_blank');
        
    });

    // Add hover event listener to the score plot
    scoreDiv.on('plotly_hover', function(eventData) {
        const point = eventData.points[0];
        const x = point.x;
        const y = scoreBoard[point.pointNumber]['Accuracy'];
        Plotly.Fx.hover('plotDiv', {xval: x, yval: y});
    });

    // Add unhover event listener to the score plot
    scoreDiv.on('plotly_unhover', function(eventData) {Plotly.Fx.unhover('plotDiv')});

});
