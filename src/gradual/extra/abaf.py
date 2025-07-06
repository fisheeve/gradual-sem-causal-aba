"""
Copyright <2023-2024> <Tuomo Lehtonen, University of Helsinki>

Permission is hereby granted, free of charge, to any person obtaining a copy of this
software and associated documentation files (the "Software"), to deal in the Software
without restriction, including without limitation the rights to use, copy, modify,
merge, publish, distribute, sublicense, and/or sell copies of the Software, and to
permit persons to whom the Software is furnished to do so, subject to the following
conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED,
INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR
PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT
OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
OTHER DEALINGS IN THE SOFTWARE.
"""

from collections import defaultdict


class ABAF:

    def __init__(self):
        self.asmpt_to_idx = dict()
        self.idx_to_asmpt = dict()
        self.atom_to_idx = dict()
        self.idx_to_atom = dict()
        self.rules = list()
        self.contraries = defaultdict(set)

        self.atom_counter = 1
        self.rule_counter = 1

    def add_asmpt(self, name):
        self.asmpt_to_idx[name] = self.atom_counter
        self.idx_to_asmpt[self.atom_counter] = name
        self.atom_to_idx[name] = self.atom_counter
        self.idx_to_atom[self.atom_counter] = name
        self.atom_counter += 1

    def add_rule(self, head, body):

        if head not in self.atom_to_idx:
            self.atom_to_idx[head] = self.atom_counter
            self.idx_to_atom[self.atom_counter] = head
            self.atom_counter += 1
        for b in body:
            if b not in self.atom_to_idx:
                self.atom_to_idx[b] = self.atom_counter
                self.idx_to_atom[self.atom_counter] = b
                self.atom_counter += 1

        self.rules.append((self.atom_to_idx[head], [self.atom_to_idx[body_elem] for body_elem in body]))

    def add_contrary(self, asmpt, contrary):
        # NOTE: assuming that asmpt already exists; check elsewhere!
        assert asmpt in self.asmpt_to_idx, f"When adding contrary, assumption {asmpt} not found in assumptions."

        if contrary not in self.atom_to_idx:
            self.atom_to_idx[contrary] = self.atom_counter
            self.idx_to_atom[self.atom_counter] = contrary
            self.atom_counter += 1

        self.contraries[self.atom_to_idx[asmpt]].add(self.atom_to_idx[contrary])
