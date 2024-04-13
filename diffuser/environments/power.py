import matlab.engine as me
import numpy as np
import matlab


def reward(s, ns, a):
    # f1, f2 = s[:2]
    # nf1, nf2 = ns[:2]

    # r4f = np.exp(-(abs(f1) + abs(f2)) * 1e2)

    # df1 = abs(f1) - abs(nf1)
    # df2 = abs(f2) - abs(nf2)
    # r4df = (df1 + df2) * 1e2

    # a1, a2 = a
    # r4p = (a1 + a2) * 1e-1

    # return r4f + r4df + r4p
    return 0


class PowerEnv:
    """
    Communication with Simulink.
    """

    def __init__(self) -> None:
        if me.find_matlab():
            session_name = me.find_matlab()[0]
            self.eng = me.connect_matlab(session_name)

        self.state_names = [
            "delt_f1",
            "delt_f2",
            # "delt_df1",
            # "delt_df2",
            "Pm1",
            "Pm2",
            # "Pe1",
            # "Pe2",
            "Pg1",
            "Pg2",
            "Ptie",
        ]

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
        # self.set("prev_action", action)
        return (
            next_state,
            reward(state, next_state, action),
            False,
            False,
        )  # terminal, timout

    def update(self):
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
        self.eval('sim("assets/env.slx")')
        self.eval(update_cmd)

    def state_vector(self):
        return np.array([self.get(k) for k in self.state_names])

    def set_state(self, state):
        for k, v in zip(self.state_names, state):
            cmd = f'{k} = timeseries([0, {v}], [0, 1]);'
            self.eval(cmd)

    def reset(self):
        self.eval("clear")
        self.set("stop_init", 0.1)
        self.set("stop_env", 1.0)
        self.set("sample", 0.1)
        self.set("prev_action", [0, 0])
        self.eval('sim("assets/init.slx")')
        return self.state_vector()

    def get_normalized_score(self, total_reward):
        return sum(self.state_vector()[:2])
