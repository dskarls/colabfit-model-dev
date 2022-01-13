import torch


class ScalarAdder(torch.nn.Module):
    def forward(self, A: torch.Tensor, B: torch.Tensor):
        return A + B


if __name__ == "__main__":

    scalar_adder = ScalarAdder()
    scalar_adder_script_compiled = torch.jit.script(scalar_adder)
    scalar_adder_script_compiled.save("add-two-scalars.pt")
