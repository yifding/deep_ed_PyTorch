function first_letter_to_uppercase(s)
  return s:sub(1,1):upper() .. s:sub(2)
end


test_line_4 = '<doc id="12" url="http://en.wikipedia.org/wiki?curid=12" title="Anarchism">'

--startoff, endoff = test_line_4:find('<doc id="')
--print(startoff, endoff)

--print(test_line_4:sub(6, 8))


function trim1(s)
   return (s:gsub("^%s*(.-)%s*$", "%1"))
end

--print('AAA' .. trim1('\n   asdf   \n') .. 'AAA')
s = '\n   asdf   \n'
print(s:gsub('\n', 'aa'))
print(string.gsub(s, '\n', 'aa'))

a = 'æ'
if a then
    print('122')
end

if 'æ' then
    print('122')
end

print(first_letter_to_uppercase('æ'))
print(first_letter_to_uppercase('a'))