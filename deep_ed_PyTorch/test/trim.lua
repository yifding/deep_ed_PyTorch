function trim1(s)
   return (s:gsub("^%s*(.-)%s*$", "%1"))
end

print('AAA' .. trim1('\nasdf\n') .. 'AAA')
print('AAA' .. trim1('\tasdf\t') .. 'AAA')
print('AAA' .. trim1('\rasdf\r') .. 'AAA')
print('AAA' .. trim1(' asdf ') .. 'AAA')
print('AAA' .. trim1('\vasdf\v') .. 'AAA')
print('AAA' .. trim1('\fasdf\f') .. 'AAA')