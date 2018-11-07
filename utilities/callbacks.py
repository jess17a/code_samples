save_datasource_code = """
    var array = objArray.attributes.data;
    var str = '';
    var line = '';
    objKeys = objArray.attributes.column_names
    var placeholder_key = objKeys[0]

    for (var i = 0; i < objKeys.length; i++) {
        var value = objKeys[i] + "";
        line += '"' + value.replace(/"/g, '""') + '",';
    }

    line = line.slice(0, -1);
    str += line + '\\r\\n';

    for (var i = 0; i < array[placeholder_key].length; i++) {
        var line = '';

        for (var j = 0; j < objKeys.length; j++) {
            var index = objKeys[j]
            var value = array[index][i] + "";
            line += '"' + value.replace(/"/g, '""') + '",';
        }
        line = line.slice(0, -1);
        str += line + '\\r\\n';
    }
    var blob = new Blob([str], { type: 'text/csv;charset=utf-8;' });
    if (navigator.msSaveBlob) { // IE 10+
        navigator.msSaveBlob(blob, filename);
    } else {
        var link = document.createElement("a");
        if (link.download !== undefined) { // feature detection
            // Browsers that support HTML5 download attribute
            var url = URL.createObjectURL(blob);
            link.setAttribute("href", url);
            link.setAttribute("download", objArray.name);
            link.style.visibility = 'hidden';
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
        }
    }
    """

print_div_code = """
var divContents = document.getElementById('div_to_print').innerHTML
var printWindow = window.open('', '', 'height=1,width=1');
printWindow.document.write('<html><head><title>Scholar Reports</title>');
printWindow.document.write('</head><body >');
printWindow.document.write(divContents);
printWindow.document.write('</body></html>');
printWindow.document.close();
printWindow.print();
printWindow.close()
"""


rightsize_plots_code = '''
function findAncestor (el, cls) {
    while ((el = el.parentElement) && !el.classList.contains(cls));
    return el;
}

var canvases = document.getElementsByTagName('canvas')
for (i = 0; i < canvases.length; i++) {
    var ancestor = findAncestor(canvases[i], 'bk-plot-layout')
    canvases[i].style['width'] = ancestor.style['width']
    canvases[i].style['height'] = ancestor.style['height']

}
'''

window_resize_code = '''
window.dispatchEvent(new Event('resize'));
'''