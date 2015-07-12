function theta = rtrbm_stack2params(stack)
theta = [stack.w_vh(:); stack.w_hh(:);stack.a;stack.b];
end

