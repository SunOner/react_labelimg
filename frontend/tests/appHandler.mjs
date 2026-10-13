import assert from 'node:assert/strict'
import fs from 'node:fs'
import ts from 'typescript'

// Execute a real App handler with injected dependencies to cover its wiring.
export function appHandler(name, context) {
  const source = fs.readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8')
  const file = ts.createSourceFile('App.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  let initializer
  function visit(node) {
    if (ts.isVariableDeclaration(node) && node.name.getText(file) === name) initializer = node.initializer.getText(file)
    ts.forEachChild(node, visit)
  }
  visit(file)
  assert.ok(initializer, `App handler ${name} exists`)
  const code = ts.transpileModule(`const ${name} = ${initializer};`, {
    compilerOptions: { target: ts.ScriptTarget.ES2022 },
  }).outputText
  return new Function('context', `with(context) { ${code}; return ${name}; }`)(context)
}

export function appEffect(marker, context) {
  const source = fs.readFileSync(new URL('../src/App.tsx', import.meta.url), 'utf8')
  const file = ts.createSourceFile('App.tsx', source, ts.ScriptTarget.Latest, true, ts.ScriptKind.TSX)
  let callback
  function visit(node) {
    if (ts.isCallExpression(node) && node.expression.getText(file) === 'useEffect' && node.arguments[0].getText(file).includes(marker)) callback = node.arguments[0].getText(file)
    ts.forEachChild(node, visit)
  }
  visit(file)
  assert.ok(callback, `App effect ${marker} exists`)
  const code = ts.transpileModule(`const effect = ${callback};`, { compilerOptions: { target: ts.ScriptTarget.ES2022 } }).outputText
  return new Function('context', `with(context) { ${code}; return effect; }`)(context)
}
