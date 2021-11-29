require 'nn'

num_mentions = 2
max_num_cand = 3
ent_vecs_size = 4

function new_linear_layer(out_dim)
  cmul = nn.CMul(out_dim)
  -- init weights with ones to speed up convergence
  cmul.weight = torch.ones(out_dim)
  return cmul
end

param_C_linear = new_linear_layer(ent_vecs_size)

model = nn.Sequential() -- Pairwise scores s_{ij}(y_i, y_j)
            :add(nn.View(num_mentions * max_num_cand, ent_vecs_size)) -- e_vecs
            :add(nn.ConcatTable()
                :add(nn.Identity())
                :add(param_C_linear)
            )
            :add(nn.MM(false, true))-- s_{ij}(y_i, y_j) is s[i][y_i][j][y_j] = <e_{y_i}, C * e_{y_j}>
        -- :add(nn.View(num_mentions, max_num_cand, num_mentions, max_num_cand))
        -- :add(nn.MulConstant(2.0 / num_mentions, true))




a = torch.rand(2,3,4)
print(model:forward(a))



model_max = nn.Sequential()
            :add(nn.Max(4))
            :add(nn.MulConstant(0, false))

b = torch.rand(2,3,2,3)
-- print(b)
print(model_max:forward(b))