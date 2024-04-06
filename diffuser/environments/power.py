import matlab.engine as me
import numpy as np
import matlab


def reward(s, ns, a):
    f1, f2 = s[:2]
    nf1, nf2 = ns[:2]

    r4f = np.exp(-abs(f1 + f2) * 1e2)

    df1 = abs(f1) - abs(nf1)
    df2 = abs(f2) - abs(nf2)
    r4df = (df1 + df2) * 1e2

    a1, a2 = a
    r4p = (a1 + 1e1 * a2) * 1e-1

    return r4f + r4df + r4p


class PowerEnv:
    """
    Communication with Simulink.
    """

    def __init__(self) -> None:
        assert len(me.find_matlab()) == 1
        session_name = me.find_matlab()[0]
        self.eng = me.connect_matlab(session_name)
        self.reset()

    def set(self, k, v):
        if isinstance(v, (list, tuple)):
            v = matlab.double(v)
        self.eng.workspace[k] = v

    def get(self, k, series=True):
        if series:
            return self.eval(k + ".Data", nout=1)[-1][0]
        return self.eng.workspace[k]

    def eval(self, cmd, nout=0):
        return self.eng.eval(cmd, nargout=nout)

    def step(self, action):
        state = self.state_vector()
        self.set("action", action)
        self.update()
        next_state = self.state_vector()
        self.set("prev_action", action)
        return next_state, reward(state, next_state, action)

    def update(self):
        self.eval('sim("assets/env.slx")')
        update_cmd = """delt_f1 = env_delt_f1;
                        delt_f2 = env_delt_f2;
                        delt_df1 = env_delt_df1;
                        delt_df2 = env_delt_df2;
                        Pm1 = env_Pm1;
                        Pm2 = env_Pm2;
                        Pe1 = env_Pe1;
                        Pe2 = env_Pe2;
                        Pg1 = env_Pg1;
                        Pg2 = env_Pg2;
                        Ptie = env_Ptie;"""
        self.eval(update_cmd)

    def state_vector(self):
        return np.array(
            [
                self.get("delt_f1"),
                self.get("delt_f2"),
                self.get("delt_df1"),
                self.get("delt_df2"),
                self.get("Pm1"),
                self.get("Pm2"),
                self.get("Pe1"),
                self.get("Pe2"),
                self.get("Pg1"),
                self.get("Pg2"),
                self.get("Ptie"),
            ]
        )

    def set_state(self, state):
        pass

    def reset(self):
        self.eval("clear")
        self.set("stop_init", 0.1)
        self.set("stop_env", 0.1)
        self.set("sample", 0.1)
        self.set("prev_action", [0, 0])
        self.eval('sim("assets/init.slx")')
        return self.state_vector()

    def get_normalized_score(self):
        return sum(self.state_vector()[:2])
