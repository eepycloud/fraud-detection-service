const fs = require('node:fs');
const vm = require('node:vm');
const html = fs.readFileSync('src/dashboard.html', 'utf8');
const scripts = [...html.matchAll(/<script>([\s\S]*?)<\/script>/g)];
if (!scripts.length) throw new Error('Dashboard has no script');
for (const [index, match] of scripts.entries()) new vm.Script(match[1], {filename: 'dashboard-script-' + index + '.js'});
console.log('PASS: dashboard JavaScript syntax');
