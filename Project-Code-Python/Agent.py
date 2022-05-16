from PIL import Image, ImageChops
from ProblemSet import ProblemSet
import numpy


def tversky(img_a, img_b):
    invert_a = numpy.invert(img_a)
    invert_b = numpy.invert(img_b)
    t = numpy.sum(numpy.logical_and(invert_a, invert_b)) / (
        numpy.sum(numpy.logical_and(invert_a, invert_b)) + numpy.sum(numpy.logical_xor(invert_a, invert_b)))
    return t


class Agent:
    def __init__(self):
        self.a = None
        self.b = None
        self.c = None
        self.d = None
        self.e = None
        self.f = None
        self.g = None
        self.h = None
        self.candidates = []

    def Solve(self, problem):
        target = None
        self.a = numpy.array(Image.open(problem.figures["A"].visualFilename).convert("1"))
        self.b = numpy.array(Image.open(problem.figures["B"].visualFilename).convert("1"))
        self.c = numpy.array(Image.open(problem.figures["C"].visualFilename).convert("1"))
        self.d = numpy.array(Image.open(problem.figures["D"].visualFilename).convert("1"))
        self.e = numpy.array(Image.open(problem.figures["E"].visualFilename).convert("1"))
        self.f = numpy.array(Image.open(problem.figures["F"].visualFilename).convert("1"))
        self.g = numpy.array(Image.open(problem.figures["G"].visualFilename).convert("1"))
        self.h = numpy.array(Image.open(problem.figures["H"].visualFilename).convert("1"))
        self.candidates = [numpy.array(Image.open(problem.figures[str(x)].visualFilename).convert("1")) for
                           x in range(1, 9)]

        if tversky(self.a, self.c) > .85 and tversky(self.d, self.f) > .85:
            target = self.g
        elif tversky(numpy.logical_and(self.a, self.b), self.c) > .75 and tversky(numpy.logical_and(self.d, self.e), self.f) > .75:
            target = numpy.logical_and(self.g, self.h)
        elif tversky(numpy.logical_or(self.a, numpy.invert(self.b)), self.c) > .75 and tversky(numpy.logical_or(self.d, numpy.invert(self.e)), self.f) > .75:
            target = numpy.logical_or(self.g, numpy.invert(self.h))
        elif tversky(numpy.logical_or(self.b, numpy.invert(self.a)), self.c) > .75 and tversky(numpy.logical_or(self.e, numpy.invert(self.d)), self.f) > .75:
            target = numpy.logical_or(self.h, numpy.invert(self.g))
        elif tversky(numpy.invert(numpy.logical_xor(self.a, self.b)), self.c) > .75 and tversky(numpy.invert(numpy.logical_xor(self.d, self.e)), self.f) > .75:
            target = numpy.invert(numpy.logical_xor(self.g, self.h))
        elif tversky(numpy.logical_or(self.a, self.b), self.c) > .85 and tversky(numpy.logical_or(self.d, self.e), self.f) > .85:
            target = numpy.logical_or(self.g, self.h)

        try:
            candidate_scores = []
            for c in self.candidates:
                candidate_scores.append(tversky(target, c))
            return candidate_scores.index(max(candidate_scores)) + 1
        except TypeError:
            a_px = numpy.sum(self.a)
            b_px = numpy.sum(self.b)
            c_px = numpy.sum(self.c)
            d_px = numpy.sum(self.d)
            e_px = numpy.sum(self.e)
            f_px = numpy.sum(self.f)
            g_px = numpy.sum(self.g)
            h_px = numpy.sum(self.h)
            s_px = sorted([a_px, b_px, c_px, d_px, e_px, f_px, g_px, h_px])

            if abs(f_px / e_px - c_px / b_px) < .05 and abs(e_px / d_px - b_px / a_px) < .05:
                target = (c_px / a_px + f_px / d_px) * g_px / 2
            elif abs(h_px / e_px - g_px / d_px) < .05 and abs(e_px / b_px - d_px / a_px) < .05:
                target = (numpy.sum(self.g) / numpy.sum(self.a) + numpy.sum(self.h) / numpy.sum(self.b)) * numpy.sum(
                    self.c) / 2
            elif abs(s_px[2] - s_px[0]) < 100 and abs(s_px[5] - s_px[3]) < 100 and abs(s_px[7] - s_px[6]) < 100:
                target = (s_px[7] + s_px[6]) / 2
            elif abs(s_px[2] - s_px[0]) < 100 and abs(s_px[4] - s_px[3]) < 100 and abs(s_px[7] - s_px[5]) < 100:
                target = (s_px[4] + s_px[3]) / 2
            elif abs(s_px[1] - s_px[0]) < 100 and abs(s_px[4] - s_px[2]) < 100 and abs(s_px[7] - s_px[5]) < 100:
                target = (s_px[1] + s_px[0]) / 2

            if target:
                candidate_scores = [abs(numpy.sum(x) - target) for x in self.candidates]
                return candidate_scores.index(min(candidate_scores)) + 1
            else:
                given_list = [self.a, self.b, self.c, self.d, self.e, self.f, self.g, self.h]
                tversky_list = []
                for i in given_list:
                    temp_list = []
                    for j in given_list:
                        temp_list.append(tversky(i, j))
                    tversky_list.append(temp_list)
                k = 0
                nt = []
                for i in tversky_list:
                    if (sorted(i)[-3]) > .8:
                        k += 1
                    else:
                        nt.append(i.index(1))
                if k >= 6:
                    target = given_list[nt[0]]
                elif tversky(self.b, self.d) > .6:
                    bd = numpy.logical_or(self.b, self.d)
                    target = numpy.logical_and(bd, self.e)
                ab = numpy.logical_or(self.a, self.b)
                ac = numpy.logical_or(self.a, self.c)
                bc = numpy.logical_or(self.b, self.c)
                abc = numpy.logical_or(ab, self.c)
                bf = numpy.logical_or(self.b, self.f)
                bg = numpy.logical_or(self.b, self.g)
                fg = numpy.logical_or(self.f, self.g)
                bfg = numpy.logical_or(bf, self.g)
                ae = numpy.logical_or(self.a, self.e)
                if tversky(ab, ac) > .85 and tversky(ac, bc) > .85 and tversky(ab, bc) > .85:
                    gh = numpy.logical_or(self.g, self.h)
                    g_1 = numpy.logical_and(abc, self.g)
                    h_1 = numpy.logical_and(abc, self.h)
                    paired = [0, 0, 0]
                    first_row = [self.a, self.b, self.c]
                    for i in [g_1, h_1]:
                        coef_list = []
                        for j in first_row:
                            coef_list.append(tversky(i, j))
                        paired[coef_list.index(max(coef_list))] = 1
                    temp_target = numpy.logical_or(first_row[paired.index(0)], numpy.invert(abc))
                    target = numpy.logical_and(temp_target, gh)
                # Image.fromarray(bf).show()
                # Image.fromarray(bg).show()
                # Image.fromarray(fg).show()
                elif numpy.sum(bfg) < 32000:
                    #Image.fromarray(bfg).show()
                    #Image.fromarray(ae).show()
                    print("VALID")
                    raw_a = numpy.invert(numpy.logical_xor(self.a, ae))
                    raw_e = numpy.invert(numpy.logical_xor(self.e, ae))
                    raw_b = numpy.invert(numpy.logical_xor(self.b, bfg))
                    raw_f = numpy.invert(numpy.logical_xor(self.f, bfg))
                    raw_g = numpy.invert(numpy.logical_xor(self.g, bfg))
                    paired = [0, 0, 0]
                    rbfg_l = [raw_b, raw_f, raw_g]
                    for i in [raw_a, raw_e]:
                        temp_list = []
                        for j in rbfg_l:
                            temp_list.append(tversky(i, j))
                        paired[temp_list.index(max(temp_list))] = 1
                    target = numpy.logical_and(rbfg_l[paired.index(0)], ae)

                try:
                    candidate_scores = []
                    for c in self.candidates:
                        candidate_scores.append(tversky(target, c))
                    return candidate_scores.index(max(candidate_scores)) + 1
                except TypeError:
                    return False


if __name__ == '__main__':
    import os
    a = Agent()
    ps = ProblemSet('Basic Problems D')
    for pb in ps.problems[:]:
        answer = a.Solve(pb)
        with open("Problems" + os.sep + ps.name + os.sep + pb.name + os.sep + "ProblemAnswer.txt") as f:
            solution = f.readline()
            f.close()
        print(f'{pb.name}: {answer} - {int(solution)}')
        print()