path = '/scratch365/yding4/EL_resource/data/deep_ed_data/basic_data/test_datasets/wned-datasets/ace2004/ace2004.xml'
annotations, _ = io.open(path)

line = annotations:read()
num_line = 0
while line do
    num_line = num_line + 1
    if line:find('\n') then
        print('\n exist in lua read')
    end
    len_line = string.len(line)
    print(line:sub(1, 3))
    print(line:sub(len_line-3, len_line))
    line = annotations:read()
    if num_line >= 1 then
        break
    end
end




out_file = 'test_lua.log'
ouf = assert(io.open(out_file, "w"))
ouf:write('L\t\n')
ouf:write('L\r\n')
ouf:write('L\f\n')
ouf:write('L\v\n')
ouf:write('L \n')
ouf:flush()
io.close(ouf)

reader, _ = io.open(out_file)
line = reader:read()

while line do
    print(string.len(line))
    print(line)
    line = reader:read()
end