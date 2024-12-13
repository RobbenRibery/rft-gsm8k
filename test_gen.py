from generation import _socre_equations  

if __name__ == "__main__":

    equations = [
        "1 + 2 = 3",
        "1. 1 + 2 = 3",
        "1. 2 + 1 = 3",
    ]

    max_idx, min_idx, diff = _socre_equations(equations, beta_1=0.5)
    assert max_idx == 2 
    assert min_idx == 1


    equations = [
        "1 (apple) + 2 (orange) = 3 (strange thing)",
        "1. 1 + 2 = 3",
        "1. apple 2 + organge 1 = ?3 ",
    ]

    max_idx, min_idx, diff =_socre_equations(equations, beta_1=0.5,)
    assert max_idx == 0 
    assert min_idx == 1