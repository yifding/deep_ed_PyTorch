require 'nn'

function correct_type(data)
  if opt.type == 'float' then return data:float()
  elseif opt.type == 'double' then return data:double()
  elseif string.find(opt.type, 'cuda') then return data:cuda()
  else print('Unsuported type')
  end
end



-- unit test parameter

cmd = torch.CmdLine()
cmd:option('-type', 'double', 'type: double | float | cuda | cudacudnn')
cmd:option('-batch_size', 7, 'mini-batch size (1 = pure stochastic)')
cmd:option('-num_words_per_ent', 2, 'num positive words per entity per iteration.')
cmd:option('-num_neg_words', 25, 'num negative words in the partition function.')
cmd:option('-loss', 'nce', 'nce | neg | is | maxm')
cmd:option('-init_vecs_title_words', false, 'whether the entity embeddings should be initialized with the average of title word embeddings. Helps to speed up convergence speed of entity embeddings learning.')
opt = cmd:parse(arg or {})
word_vecs_size = 5
ent_vecs_size = word_vecs_size
lookup_ent_vecs = nn.LookupTable(100, ent_vecs_size)


-- model test

model = nn.Sequential()
  :add(nn.ConcatTable()
    :add(nn.Sequential()
      :add(nn.SelectTable(1))
      :add(nn.SelectTable(2)) -- ctxt words vectors
      :add(nn.Normalize(2))
      :add(nn.View(opt.batch_size, opt.num_words_per_ent * opt.num_neg_words, ent_vecs_size)))
    :add(nn.Sequential()
      :add(nn.SelectTable(3))
      :add(nn.SelectTable(1))
      :add(lookup_ent_vecs) -- entity vectors
      :add(nn.Normalize(2))
      :add(nn.View(opt.batch_size, 1, ent_vecs_size))))
  :add(nn.MM(false, true))
  :add(nn.View(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words))

-- prepare input

inputs = {}
inputs[1] = {}
inputs[1][1] = correct_type(torch.ones(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words)) -- ctxt words

inputs[2] = {}
inputs[2][1] = correct_type(torch.ones(opt.batch_size * opt.num_words_per_ent)) -- ent wiki words

inputs[3] = {}
inputs[3][1] = correct_type(torch.ones(opt.batch_size)) -- ent th ids
inputs[3][2] = torch.ones(opt.batch_size) -- ent wikiids

-- ctxt word vecs
inputs[1][2] = correct_type(torch.ones(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words, word_vecs_size))

inputs[1][3] = correct_type(torch.randn(opt.batch_size * opt.num_words_per_ent * opt.num_neg_words))


outputs = model:forward(inputs)
assert(outputs:size(1) == opt.batch_size * opt.num_words_per_ent and
    outputs:size(2) == opt.num_neg_words)
print('FWD success!')

model:backward(inputs, correct_type(torch.randn(opt.batch_size * opt.num_words_per_ent, opt.num_neg_words)))
print('BKWD success!')

