import gym
from gym import spaces
import numpy as np
import casadi as ca
from tqdm.auto import tqdm
import random

class MBRGymEnv(gym.Env):
    def __init__(self, wOCI=0.1):
        super(MBRGymEnv, self).__init__()
        self.wOCI = wOCI
        self.t = 0

        self.mu_H = 4
        self.mu_A = 0.5
        self.b_H = 0.3
        self.b_A = 0.05  # 1/day
        self.k_a = 0.05  # m3/(gCOD*day)
        self.k_h = 3  # gCOD/(gCOD*day)
        self.K_S = 10
        self.K_OH = 0.2
        self.K_NO = 0.5
        self.K_NH = 1
        self.K_OA = 0.4  # self.g/m3
        self.K_X = 0.1  # gCODp/gCOD
        self.eta_g = 0.8
        self.eta_h = 0.8  # unitless
        self.Y_H = 0.67  # gCOD/gCOD
        self.Y_A = 0.24  # gCOD/self.g
        self.i_XB = 0.08
        self.i_XP = 0.06  # gN/gCOD
        self.f_P = 0.08  # unitless

        # Aeration model parameters
        ## Table 2 in Guo2020: BSM-MBR model and the membrane fouling model parameters
        ## used in oxygen transfer rate through aeration model
        self.beta = 0.95
        self.F = {1: 0, 2: 0, 3: 0.7, 4: 0.7, 5: 0.9}  # unitless
        self.g = 9.81  # N/kg
        self.O_Am = 0.232
        self.O_Av = 0.21  # unitless
        self.P_atm = 101325  # Pa
        self.rho_A = 1200  # self.g/m3
        self.rho_S = 1000  # kg/m3
        self.SOTE = {1: 0, 2: 0, 3: 0.06, 4: 0.06, 5: 0.02}  # 1/m
        self.T = 15  # deg C
        self.yi = {1: 0, 2: 0, 3: 5, 4: 5, 5: 3.5}
        self.h = 5  # m
        self.phi = 1.024  # unitless
        self.omega = {1: 0, 2: 0, 3: 0.083, 4: 0.083, 5: 0.05}  # m
        self.qO2_a    = np.zeros([5])
        # self.QA3 = 4250*24
        # self.QA4 = 2250*24

        # Tank Volumes (Unit: m3)
        self.Vol = {1: 1500, 2: 1500, 3: 1500, 4: 1500, 5: 1500}
        # Flow rates (Unit: m3/day)
        self.Q_feed = 21477
        self.Q_int = 55338
        self.Q_r = 55338
        self.Q_w = 200
        # fouling parameters:
        self.day_s = 1 * 24 * 60 * 60 #convert day to second
        self.day_min = 1 * 24 * 60
        self.m3_L = 1000
        self.kg_g = 1000
        self.N = 1
        self.f = 3.45  # wet to dry ratio
        self.pi = 3.14159
        self.Rm = 1e11
        self.Rapparent = 0
        self.Rpapparent = 0
        self.Rsfapparent = 0
        self.Rscapparent = 0
        self.Rp = 0

        self.area = 71500  # membrane area m2
        self.J0 = (self.Q_feed - self.Q_w) / self.area

        self.mu_wPas = 1.02e-3
        self.mu_w = 1.02e-3 * self.day_s
        self.mu_PaDay = 1.02e-3 / self.day_s
        self.Vtf_ = 0

        self.stickiness2 = 0.5
        self.stickiness1 = 0.2 * self.stickiness2
        self.K1 = 4e-6
        self.rp = 2e11
        self.rsf = 1e15
        self.rsc = 1e15
        self.WetDensity = 1.06e3
        self.beta_f = 3.5e-4
        self.gamma1 = 2.5e-5 * self.day_s
        self.SDensity = 1000
        self.g_m_day = 9.81 * self.day_s * self.day_s

        self.N_Tanks = 5
        self.N_Vars = 13 * self.N_Tanks + 1

        self.C_star_20 = -0.00006588 * 20**3 + 0.007311 * 20**2 - 0.3825 * 20 + 13.89
        self.C_star_T = -0.00006588 * self.T**3 + 0.007311 * self.T**2 - 0.3825 * self.T + 13.89

        self.tf = 13.5*60 # s
        self.tc = 1.5*60
        self.cycle_time = int(self.tf + self.tc)
        self.t_operate = 14 # day
        # circle = int(96 * self.t_operate)

        self.tf_d = self.tf / self.day_s # min->day
        self.tc_d = self.tc / self.day_s # min->day
        self.tp_d = self.tf_d + self.tc_d
        self.dt = 1 / self.day_s # s->day
        self.steps = int(self.tp_d/self.dt)

        self.ksc = self.f / self.WetDensity
        # Define action space and observation space
        # Example: actions are flow rates, observations are states
        self.action_low =  np.array([4000, 2000, 5000, 0])
        self.action_high = np.array([4500, 2500, 38012, 32180.0*5]) #KLA5 Q_recycle

        # self.action_low =  np.array([4000, 2000, 5000, 0])
        # self.action_high = np.array([4500, 2500, 22000, 32180.0*3]) #KLA5 Q_recycle

        # self.action_low = np.array([4000, 2000, 15015, 0])
        # self.action_high = np.array([4500,2500,38012, 32180.0 * 3])  # KLA5 Qa

        self.action_const = np.array([21450, 55338])
        self.action_space = spaces.Box(low=self.action_low, high=self.action_high, dtype=np.float32)
        self.action_sample_period = 50
        self.action_slope_change = 5
        

        yset1 = 1.48 #x22 setpoint 
        yset2 = 8.11 #x60 setpoint
        
        self.xs = np.array([yset1,yset2])

        # Initialize states and parameters
        self.state = np.zeros(self.N_Vars)
        high = np.ones_like(self.state)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)
        self.disturbance = np.loadtxt('envs/Inf_dry_constQ.txt') 
        self.disturbance = np.hstack((self.disturbance[:, -1].reshape(-1, 1), self.disturbance[:, 1:-1]))
        self.disturbance = np.hstack((self.disturbance, np.zeros((self.disturbance.shape[0], 1))))
        self.time_step = 0
        # self.wwtp_casadi = mpc.getCasadiFunc(model.ode_15, [self.N_Vars, self.action_low.shape[0], self.disturbance.shape[1]], ["x","u","p"], funcname="f", Delta=1)
        

        high = np.array(np.ones(self.N_Vars))*np.inf
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        self.all_state = 66
        self.pred_index = set(np.arange(self.all_state))
        self.cont_index = set([ 0,  13,  26,  39,  52])
        self.pred_tracking = np.array(list(self.pred_index-self.cont_index))
        self.state_use = self.pred_tracking.shape[0]

        self.state_tracking  = np.array([59,21])
        self.state_tracking_c= np.array([54,19])
        self.draw_state      = np.array([1,20,21,22,53,54,55,66])-1
        self.draw_state_c    = np.array([0,18,19,20,48,49,50,61])-1
        self.draw_state_num  = self.draw_state.shape[0]

        self._ini_ode_1s()
        self._ini_ode_15()
        self._ini_econ_cost()
        self._ini_step()
        
    def _ini_ode_1s(self):
        x = ca.MX.sym('x', self.N_Vars)
        u = ca.MX.sym('u', self.action_low.shape[0])
        p = ca.MX.sym('p', self.disturbance.shape[1])
        t = ca.MX.sym('t', 1)
        Q0 = p[0]
        Z0 = p[1:] 
    
    # Input  2 variables
        
        QA3 = u[0]*24
        QA4 = u[1]*24
        QA5 = u[2]*24

        Qa = u[3]
        Qr = u[3]
        
        x1 = x[0] 
        x2 = x[1] 
        x3 = x[2] 
        x4 = x[3] 
        x5 = x[4] 
        x6 = x[5]
        x7 = x[6] 
        x8 = x[7] 
        x9 = x[8] 
        x10 = x[9] 
        x11 = x[10] 
        x12 = x[11]
        x13 = x[12] 
        x14 = x[13] 
        x15 = x[14] 
        x16 = x[15] 
        x17 = x[16]
        x18 = x[17] 
        x19 = x[18] 
        x20 = x[19] 
        x21 = x[20] 
        x22 = x[21]
        x23 = x[22] 
        x24 = x[23] 
        x25 = x[24] 
        x26 = x[25] 
        x27 = x[26] 
        x28 = x[27] 
        x29 = x[28] 
        x30 = x[29] 
        x31 = x[30] 
        x32 = x[31]
        x33 = x[32] 
        x34 = x[33] 
        x35 = x[34] 
        x36 = x[35] 
        x37 = x[36]
        x38 = x[37] 
        x39 = x[38] 
        x40 = x[39] 
        x41 = x[40] 
        x42 = x[41]
        x43 = x[42] 
        x44 = x[43] 
        x45 = x[44] 
        x46 = x[45] 
        x47 = x[46]
        x48 = x[47] 
        x49 = x[48] 
        x50 = x[49] 
        x51 = x[50] 
        x52 = x[51]
        x53 = x[52] 
        x54 = x[53] 
        x55 = x[54] 
        x56 = x[55] 
        x57 = x[56]
        x58 = x[57] 
        x59 = x[58] 
        x60 = x[59] 
        x61 = x[60] 
        x62 = x[61] 
        x63 = x[62] 
        x64 = x[63] 
        x65 = x[64] 
        x66 = x[65] #Msf
        
    
        Z01 = Z0[0]
        Z02 = Z0[1]
        Z03 = Z0[2]
        Z04 = Z0[3]
        Z05 = Z0[4]
        Z06 = Z0[5]
        Z07 = Z0[6]
        Z08 = Z0[7]
        Z09 = Z0[8]
        Z010 = Z0[9]
        Z011 = Z0[10]
        Z012 = Z0[11]
        Z013 = Z0[12]
        Ms_ = Z0[13]

        dxdt2 = [
        #Tank 1
                            1/self.Vol[1]*(Q0*Z01+Qa*x40-(Q0+Qa)*x1),
                            1/self.Vol[1]*(Q0*Z02+Qa*x41-(Q0+Qa)*x2) \
                                - 1/self.Y_H*(self.mu_H*x2/(self.K_S+x2)*x8/(x8+self.K_OH)*x5) - 1/self.Y_H*(self.mu_H*x2/(self.K_S+x2)*self.K_OH/(x8+self.K_OH)*x9/(self.K_NO+x9)*self.eta_g*x5) + self.k_h*x4*(x8/(x8+self.K_OH)+self.eta_h*self.K_OH*x9/(x8+self.K_OH)/(x9+self.K_NO))/(self.K_X+x4/x5),
                            1/self.Vol[1]*(Q0*Z03+Qa*x42-(Q0+Qa)*x3),
                            1/self.Vol[1]*(Q0*Z04+Qa*x43-(Q0+Qa)*x4) \
                                + (1-self.f_P)*self.b_H*x5 + (1-self.f_P)*self.b_A*x6 - self.k_h*(x4/x5)/(self.K_X+x4/x5)*(x8/(x8+self.K_OH) + self.eta_h*self.K_OH/(x8+self.K_OH)*x9/(x9+self.K_NO))*x5,
                            1/self.Vol[1]*(Q0*Z05+Qa*x44-(Q0+Qa)*x5) \
                                + (self.mu_H*x2/(self.K_S+x2)*x8/(x8+self.K_OH)*x5) + (self.mu_H*x2/(self.K_S+x2)*self.K_OH/(x8+self.K_OH)*x9/(self.K_NO+x9)*self.eta_g*x5) - self.b_H*x5,
                            1/self.Vol[1]*(Q0*Z06+Qa*x45-(Q0+Qa)*x6) \
                                + self.mu_A*x10/(self.K_NH+x10)*x8/(self.K_OA+x8)*x6 - self.b_A*x6,
                            1/self.Vol[1]*(Q0*Z07+Qa*x46-(Q0+Qa)*x7) \
                                + self.f_P*self.b_H*x5 + self.f_P*self.b_A*x6,
                            1/self.Vol[1]*(Q0*Z08+Qa*x47-(Q0+Qa)*x8) \
                                - (1-self.Y_H)/self.Y_H*(self.mu_H*x2/(self.K_S+x2)*x8/(x8+self.K_OH)*x5) - (4.57-self.Y_A)/self.Y_A*self.mu_A*x10/(self.K_NH+x10)*x8/(self.K_OA+x8)*x6,
                            1/self.Vol[1]*(Q0*Z09+Qa*x48-(Q0+Qa)*x9) \
                                - (1-self.Y_H)/(2.86*self.Y_H)*(self.mu_H*x2/(self.K_S+x2)*self.K_OH/(x8+self.K_OH)*x9/(self.K_NO+x9)*self.eta_g*x5) + 1/self.Y_A*self.mu_A*x10/(self.K_NH+x10)*x8/(self.K_OA+x8)*x6,
                            1/self.Vol[1]*(Q0*Z010+Qa*x49-(Q0+Qa)*x10) \
                                - self.i_XB*(self.mu_H*x2/(self.K_S+x2)*x8/(x8+self.K_OH)*x5) - self.i_XB*(self.mu_H*x2/(self.K_S+x2)*self.K_OH/(x8+self.K_OH)*x9/(self.K_NO+x9)*self.eta_g*x5) + (-self.i_XB-1/self.Y_A)*self.mu_A*x10/(self.K_NH+x10)*x8/(self.K_OA+x8)*x6 + self.k_a*x11*x5,
                            1/self.Vol[1]*(Q0*Z011+Qa*x50-(Q0+Qa)*x11) \
                                - self.k_a*x11*x5 + x12/x4*self.k_h*(x4/x5)/(self.K_X+x4/x5)*(x8/(x8+self.K_OH)+self.eta_h*self.K_OH/(x8+self.K_OH)*x9/(x9+self.K_NO))*x5,
                            1/self.Vol[1]*(Q0*Z012+Qa*x51-(Q0+Qa)*x12) \
                                + (self.i_XB-self.f_P*self.i_XP)*self.b_H*x5 + (self.i_XB-self.f_P*self.i_XP)*self.b_A*x6 - x12/x4*self.k_h*(x4/(x5*self.K_X+x4))*(x8/(x8+self.K_OH)+self.eta_h*self.K_OH/(x8+self.K_OH)*x9/(x9+self.K_NO))*x5,
                            1/self.Vol[1]*(Q0*Z013+Qa*x52-(Q0+Qa)*x13) \
                                - self.i_XB/14*(self.mu_H*x2/(self.K_S+x2)*x8/(x8+self.K_OH)*x5) + ((1-self.Y_H)/(14*2.86*self.Y_H)-self.i_XB/14)*(self.mu_H*x2/(self.K_S+x2)*self.K_OH/(x8+self.K_OH)*x9/(self.K_NO+x9)*self.eta_g*x5) + (-self.i_XB/14-1/(7*self.Y_A))*self.mu_A*x10/(self.K_NH+x10)*x8/(self.K_OA+x8)*x6 + 1/14*self.k_a*x11*x5,
                    #Tank2
                    1/self.Vol[2]*(Q0+Qa)*(x1-x14),
                    1/self.Vol[2]*(Q0+Qa)*(x2-x15) \
                        - 1/self.Y_H*(self.mu_H*x15/(self.K_S+x15)*x21/(x21+self.K_OH)*x18) - 1/self.Y_H*(self.mu_H*x15/(self.K_S+x15)*self.K_OH/(x21+self.K_OH)*x22/(self.K_NO+x22)*self.eta_g*x18) + self.k_h*(x17/x18)/(self.K_X+x17/x18)*(x21/(x21+self.K_OH)+self.eta_h*self.K_OH/(x21+self.K_OH)*x22/(x22+self.K_NO))*x18,
                    1/self.Vol[2]*(Q0+Qa)*(x3-x16),
                    1/self.Vol[2]*(Q0+Qa)*(x4-x17) \
                        + (1-self.f_P)*self.b_H*x18 + (1-self.f_P)*self.b_A*x19 - self.k_h*(x17/x18)/(self.K_X+x17/x18)*(x21/(x21+self.K_OH) + self.eta_h*self.K_OH/(x21+self.K_OH)*x22/(x22+self.K_NO))*x18,
                    1/self.Vol[2]*(Q0+Qa)*(x5-x18) \
                        + (self.mu_H*x15/(self.K_S+x15)*x21/(x21+self.K_OH)*x18) + (self.mu_H*x15/(self.K_S+x15)*self.K_OH/(x21+self.K_OH)*x22/(self.K_NO+x22)*self.eta_g*x18) - self.b_H*x18,
                    1/self.Vol[2]*(Q0+Qa)*(x6-x19) \
                        + self.mu_A*x23/(self.K_NH+x23)*x21/(self.K_OA+x21)*x19 - self.b_A*x19,
                    1/self.Vol[2]*(Q0+Qa)*(x7-x20) \
                        + self.f_P*self.b_H*x18 + self.f_P*self.b_A*x19,
                    1/self.Vol[2]*(Q0+Qa)*(x8-x21) \
                        - (1-self.Y_H)/self.Y_H*(self.mu_H*x15/(self.K_S+x15)*x21/(x21+self.K_OH)*x18) - (4.57-self.Y_A)/self.Y_A*self.mu_A*x23/(self.K_NH+x23)*x21/(self.K_OA+x21)*x19,
                    1/self.Vol[2]*(Q0+Qa)*(x9-x22) \
                        - (1-self.Y_H)/(2.86*self.Y_H)*(self.mu_H*x15/(self.K_S+x15)*self.K_OH/(x21+self.K_OH)*x22/(self.K_NO+x22)*self.eta_g*x18) + 1/self.Y_A*self.mu_A*x23/(self.K_NH+x23)*x21/(self.K_OA+x21)*x19,
                    1/self.Vol[2]*(Q0+Qa)*(x10-x23) \
                        - self.i_XB*(self.mu_H*x15/(self.K_S+x15)*x21/(x21+self.K_OH)*x18) - self.i_XB*(self.mu_H*x15/(self.K_S+x15)*self.K_OH/(x21+self.K_OH)*x22/(self.K_NO+x22)*self.eta_g*x18) + (-self.i_XB-1/self.Y_A)*self.mu_A*x23/(self.K_NH+x23)*x21/(self.K_OA+x21)*x19 + self.k_a*x24*x18,
                    1/self.Vol[2]*(Q0+Qa)*(x11-x24) \
                        - self.k_a*x24*x18 + x25/x17*self.k_h*(x17/x18)/(self.K_X+x17/x18)*(x21/(x21+self.K_OH)+self.eta_h*self.K_OH/(x21+self.K_OH)*x22/(x22+self.K_NO))*x18,
                    1/self.Vol[2]*(Q0+Qa)*(x12-x25) \
                        + (self.i_XB-self.f_P*self.i_XP)*self.b_H*x18 + (self.i_XB-self.f_P*self.i_XP)*self.b_A*x19 - x25/x17*self.k_h*(x17/x18)/(self.K_X+x17/x18)*(x21/(x21+self.K_OH)+self.eta_h*self.K_OH/(x21+self.K_OH)*x22/(x22+self.K_NO))*x18,
                    1/self.Vol[2]*(Q0+Qa)*(x13-x26) \
                        - self.i_XB/14*(self.mu_H*x15/(self.K_S+x15)*x21/(x21+self.K_OH)*x18) + ((1-self.Y_H)/(14*2.86*self.Y_H)-self.i_XB/14)*(self.mu_H*x15/(self.K_S+x15)*self.K_OH/(x21+self.K_OH)*x22/(self.K_NO+x22)*self.eta_g*x18) + (-self.i_XB/14-1/(7*self.Y_A))*self.mu_A*x23/(self.K_NH+x23)*x21/(self.K_OA+x21)*x19 + 1/14*self.k_a*x24*x18,

                    #Tank3 (0.75*(x29+x30+x31+x32+x33))
                    1/self.Vol[3]*((Q0+Qa)*x14 +Qr*x53 -(Q0+Qa+Qr)*x27),
                    1/self.Vol[3]*((Q0+Qa)*x15+Qr*x54-(Q0+Qa+Qr)*x28  ) \
                        - 1/self.Y_H*(self.mu_H*x28/(self.K_S+x28)*x34/(x34+self.K_OH)*x31) - 1/self.Y_H*(self.mu_H*x28/(self.K_S+x28)*self.K_OH/(x34+self.K_OH)*x35/(self.K_NO+x35)*self.eta_g*x31) + self.k_h*(x30/x31)/(self.K_X+x30/x31)*(x34/(x34+self.K_OH)+self.eta_h*self.K_OH/(x34+self.K_OH)*x35/(x35+self.K_NO))*x31,
                    1/self.Vol[3]*((Q0+Qa)*x16+Qr* x55-(Q0+Qa+Qr)*x29  ),
                    1/self.Vol[3]*((Q0+Qa)*x17+Qr*x56-(Q0+Qa+Qr)*x30  ) \
                        + (1-self.f_P)*self.b_H*x31 + (1-self.f_P)*self.b_A*x32 - self.k_h*(x30/x31)/(self.K_X+x30/x31)*(x34/(x34+self.K_OH)+self.eta_h*self.K_OH/(x34+self.K_OH)*x35/(x35+self.K_NO))*x31,
                    1/self.Vol[3]*((Q0+Qa)*x18+Qr* x57-(Q0+Qa+Qr)*x31  ) \
                        + (self.mu_H*x28/(self.K_S+x28)*x34/(x34+self.K_OH)*x31) + (self.mu_H*x28/(self.K_S+x28)*self.K_OH/(x34+self.K_OH)*x35/(self.K_NO+x35)*self.eta_g*x31) - self.b_H*x31,
                    1/self.Vol[3]*((Q0+Qa)*x19+Qr* x58-(Q0+Qa+Qr)*x32  ) \
                        + self.mu_A*x36/(self.K_NH+x36)*x34/(self.K_OA+x34)*x32 - self.b_A*x32,
                    1/self.Vol[3]*((Q0+Qa)*x20+Qr* x59-(Q0+Qa+Qr)*x33  ) \
                        +  self.f_P*self.b_H*x31 + self.f_P*self.b_A*x32,
                    1/self.Vol[3]*((Q0+Qa)*x21+Qr*x60 - (Q0+Qa+Qr)*x34 \
                        + QA3*self.rho_A*self.O_Am*((ca.exp(-self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000)*(2*self.C_star_20*self.P_atm*self.phi**20*ca.exp(self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000) - 2*self.phi**self.T*((1/self.phi**(2*self.T)*(4*self.C_star_20**2*self.P_atm**2*self.phi**40*ca.exp(2*self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000) + self.C_star_T**2*self.F[3]**2*self.P_atm**2*self.SOTE[3]**2*self.beta**2*self.phi**(2*self.T)*self.yi[3]**2 + 4*self.F[3]**2*self.O_Av**2*self.P_atm**2*self.SOTE[3]**2*self.phi**(2*self.T)*x34**2*self.yi[3]**2 + 2*self.C_star_T**2*self.F[3]**2*self.O_Av*self.P_atm**2*self.SOTE[3]**2*self.beta**2*self.phi**(2*self.T)*self.yi[3]**2 + self.C_star_T**2*self.F[3]**2*self.O_Av**2*self.P_atm**2*self.SOTE[3]**2*self.beta**2*self.phi**(2*self.T)*self.yi[3]**2 - 4*self.C_star_T*self.F[3]**2*self.O_Av*self.P_atm**2*self.SOTE[3]**2*self.beta*self.phi**(2*self.T)*x34*self.yi[3]**2 - 4*self.C_star_T*self.F[3]**2*self.O_Av**2*self.P_atm**2*self.SOTE[3]**2*self.beta*self.phi**(2*self.T)*x34*self.yi[3]**2 + self.C_star_T**2*self.F[3]**2*self.O_Av**2*self.SOTE[3]**2*self.beta**2*self.g**2*self.h**2*self.phi**(2*self.T)*self.rho_S**2*self.yi[3]**2 + 4*self.C_star_20*self.C_star_T*self.F[3]*self.P_atm**2*self.SOTE[3]*self.beta*self.phi**self.T*self.phi**20*self.yi[3]*ca.exp(self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000) + 8*self.C_star_20*self.F[3]*self.O_Av*self.P_atm**2*self.SOTE[3]*self.phi**self.T*self.phi**20*x34*self.yi[3]*ca.exp(self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000) + 2*self.C_star_T**2*self.F[3]**2*self.O_Av*self.P_atm*self.SOTE[3]**2*self.beta**2*self.g*self.h*self.phi**(2*self.T)*self.rho_S*self.yi[3]**2 + 2*self.C_star_T**2*self.F[3]**2*self.O_Av**2*self.P_atm*self.SOTE[3]**2*self.beta**2*self.g*self.h*self.phi**(2*self.T)*self.rho_S*self.yi[3]**2 - 12*self.C_star_20*self.C_star_T*self.F[3]*self.O_Av*self.P_atm**2*self.SOTE[3]*self.beta*self.phi**self.T*self.phi**20*self.yi[3]*ca.exp(self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000) - 4*self.C_star_T*self.F[3]**2*self.O_Av**2*self.P_atm*self.SOTE[3]**2*self.beta*self.g*self.h*self.phi**(2*self.T)*self.rho_S*x34*self.yi[3]**2 - 4*self.C_star_20*self.C_star_T*self.F[3]*self.O_Av*self.P_atm*self.SOTE[3]*self.beta*self.g*self.h*self.phi**self.T*self.phi**20*self.rho_S*self.yi[3]*ca.exp(self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000)))/4)**(1/2) + self.C_star_T*self.F[3]*self.P_atm*self.SOTE[3]*self.beta*self.phi**self.T*self.yi[3] + 2*self.F[3]*self.O_Av*self.P_atm*self.SOTE[3]*self.phi**self.T*x34*self.yi[3] + self.C_star_T*self.F[3]*self.O_Av*self.P_atm*self.SOTE[3]*self.beta*self.phi**self.T*self.yi[3] + self.C_star_T*self.F[3]*self.O_Av*self.SOTE[3]*self.beta*self.g*self.h*self.phi**self.T*self.rho_S*self.yi[3]))/(4*self.C_star_20*self.O_Av*self.P_atm*self.phi**20) - (self.F[3]*self.SOTE[3]*self.phi**self.T*x34*self.yi[3]*ca.exp(-self.omega[3]*0.75*(x29+x30+x31+x32+x33)/1000))/(self.C_star_20*self.phi**20))) \
                            - (1-self.Y_H)/self.Y_H*(self.mu_H*x28/(self.K_S+x28)*x34/(x34+self.K_OH)*x31) - (4.57-self.Y_A)/self.Y_A*self.mu_A*x36/(self.K_NH+x36)*x34/(self.K_OA+x34)*x32,
                    1/self.Vol[3]*((Q0+Qa)*x22+Qr*x61-(Q0+Qa+Qr)*x35  ) \
                        - (1-self.Y_H)/(2.86*self.Y_H)*(self.mu_H*x28/(self.K_S+x28)*self.K_OH/(x34+self.K_OH)*x35/(self.K_NO+x35)*self.eta_g*x31) + 1/self.Y_A*self.mu_A*x36/(self.K_NH+x36)*x34/(self.K_OA+x34)*x32,
                    1/self.Vol[3]*((Q0+Qa)*x23+Qr*x62-(Q0+Qa+Qr)*x36  ) \
                        - self.i_XB*(self.mu_H*x28/(self.K_S+x28)*x34/(x34+self.K_OH)*x31) - self.i_XB*(self.mu_H*x28/(self.K_S+x28)*self.K_OH/(x34+self.K_OH)*x35/(self.K_NO+x35)*self.eta_g*x31) + (-self.i_XB-1/self.Y_A)*self.mu_A*x36/(self.K_NH+x36)*x34/(self.K_OA+x34)*x32 + self.k_a*x37*x31,
                    1/self.Vol[3]*((Q0+Qa)*x24+Qr*x63-(Q0+Qa+Qr)*x37  ) \
                        - self.k_a*x37*x31 + x38/x30*self.k_h*(x30/x31)/(self.K_X+x30/x31)*(x34/(x34+self.K_OH)+self.eta_h*self.K_OH/(x34+self.K_OH)*x35/(x35+self.K_NO))*x31,
                    1/self.Vol[3]*((Q0+Qa)*x25+Qr* x64-(Q0+Qa+Qr)*x38  ) \
                        + (self.i_XB-self.f_P*self.i_XP)*self.b_H*x31 + (self.i_XB-self.f_P*self.i_XP)*self.b_A*x32 - x38/x30*self.k_h*(x30/x31)/(self.K_X+x30/x31)*(x34/(x34+self.K_OH)+self.eta_h*self.K_OH/(x34+self.K_OH)*x35/(self.K_NO+x35))*x31,
                    1/self.Vol[3]*((Q0+Qa)*x26+Qr*x65-(Q0+Qa+Qr)*x39  ) \
                        - self.i_XB/14*(self.mu_H*x28/(self.K_S+x28)*x34/(x34+self.K_OH)*x31) + ((1-self.Y_H)/(14*2.86*self.Y_H)-self.i_XB/14)*(self.mu_H*x28/(self.K_S+x28)*self.K_OH/(x34+self.K_OH)*x35/(self.K_NO+x35)*self.eta_g*x31) + (-self.i_XB/14-1/(7*self.Y_A))*self.mu_A*x36/(self.K_NH+x36)*x34/(self.K_OA+x34)*x32 + 1/14*self.k_a*x37*x31,

                    #Tank4
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x27-x40)  ),
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x28-x41)  ) \
                        - 1/self.Y_H*(self.mu_H*x41/(self.K_S+x41)*x47/(x47+self.K_OH)*x44) - 1/self.Y_H*(self.mu_H*x41/(self.K_S+x41)*self.K_OH/(x47+self.K_OH)*x48/(self.K_NO+x48)*self.eta_g*x44) + self.k_h*(x43/x44)/(self.K_X+x43/x44)*(x47/(x47+self.K_OH)+self.eta_h*self.K_OH/(x47+self.K_OH)*x48/(x48+self.K_NO))*x44,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x29-x42)  ),
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x30-x43)  ) \
                        + (1-self.f_P)*self.b_H*x44 + (1-self.f_P)*self.b_A*x45 - self.k_h*(x43/x44)/(self.K_X+x43/x44)*(x47/(x47+self.K_OH) + self.eta_h*self.K_OH/(x47+self.K_OH)*x48/(x48+self.K_NO))*x44,               
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x31-x44)  ) \
                        + (self.mu_H*x41/(self.K_S+x41)*x47/(x47+self.K_OH)*x44) + (self.mu_H*x41/(self.K_S+x41)*self.K_OH/(x47+self.K_OH)*x48/(self.K_NO+x48)*self.eta_g*x44) - self.b_H*x44,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x32-x45)  ) \
                        + self.mu_A*x49/(self.K_NH+x49)*x47/(self.K_OA+x47)*x45 - self.b_A*x45,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x33-x46)  ) \
                        +  self.f_P*self.b_H*x44 + self.f_P*self.b_A*x45,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x34-x47) \
                        + QA4*self.rho_A*self.O_Am*((ca.exp(-self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000)*(2*self.C_star_20*self.P_atm*self.phi**20*ca.exp(self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000) - 2*self.phi**self.T*((1/self.phi**(2*self.T)*(4*self.C_star_20**2*self.P_atm**2*self.phi**40*ca.exp(2*self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000) + self.C_star_T**2*self.F[4]**2*self.P_atm**2*self.SOTE[4]**2*self.beta**2*self.phi**(2*self.T)*self.yi[4]**2 + 4*self.F[4]**2*self.O_Av**2*self.P_atm**2*self.SOTE[4]**2*self.phi**(2*self.T)*x47**2*self.yi[4]**2 + 2*self.C_star_T**2*self.F[4]**2*self.O_Av*self.P_atm**2*self.SOTE[4]**2*self.beta**2*self.phi**(2*self.T)*self.yi[4]**2 + self.C_star_T**2*self.F[4]**2*self.O_Av**2*self.P_atm**2*self.SOTE[4]**2*self.beta**2*self.phi**(2*self.T)*self.yi[4]**2 - 4*self.C_star_T*self.F[4]**2*self.O_Av*self.P_atm**2*self.SOTE[4]**2*self.beta*self.phi**(2*self.T)*x47*self.yi[4]**2 - 4*self.C_star_T*self.F[4]**2*self.O_Av**2*self.P_atm**2*self.SOTE[4]**2*self.beta*self.phi**(2*self.T)*x47*self.yi[4]**2 + self.C_star_T**2*self.F[4]**2*self.O_Av**2*self.SOTE[4]**2*self.beta**2*self.g**2*self.h**2*self.phi**(2*self.T)*self.rho_S**2*self.yi[4]**2 + 4*self.C_star_20*self.C_star_T*self.F[4]*self.P_atm**2*self.SOTE[4]*self.beta*self.phi**self.T*self.phi**20*self.yi[4]*ca.exp(self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000) + 8*self.C_star_20*self.F[4]*self.O_Av*self.P_atm**2*self.SOTE[4]*self.phi**self.T*self.phi**20*x47*self.yi[4]*ca.exp(self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000) + 2*self.C_star_T**2*self.F[4]**2*self.O_Av*self.P_atm*self.SOTE[4]**2*self.beta**2*self.g*self.h*self.phi**(2*self.T)*self.rho_S*self.yi[4]**2 + 2*self.C_star_T**2*self.F[4]**2*self.O_Av**2*self.P_atm*self.SOTE[4]**2*self.beta**2*self.g*self.h*self.phi**(2*self.T)*self.rho_S*self.yi[4]**2 - 12*self.C_star_20*self.C_star_T*self.F[4]*self.O_Av*self.P_atm**2*self.SOTE[4]*self.beta*self.phi**self.T*self.phi**20*self.yi[4]*ca.exp(self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000) - 4*self.C_star_T*self.F[4]**2*self.O_Av**2*self.P_atm*self.SOTE[4]**2*self.beta*self.g*self.h*self.phi**(2*self.T)*self.rho_S*x47*self.yi[4]**2 - 4*self.C_star_20*self.C_star_T*self.F[4]*self.O_Av*self.P_atm*self.SOTE[4]*self.beta*self.g*self.h*self.phi**self.T*self.phi**20*self.rho_S*self.yi[4]*ca.exp(self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000)))/4)**(1/2) + self.C_star_T*self.F[4]*self.P_atm*self.SOTE[4]*self.beta*self.phi**self.T*self.yi[4] + 2*self.F[4]*self.O_Av*self.P_atm*self.SOTE[4]*self.phi**self.T*x47*self.yi[4] + self.C_star_T*self.F[4]*self.O_Av*self.P_atm*self.SOTE[4]*self.beta*self.phi**self.T*self.yi[4] + self.C_star_T*self.F[4]*self.O_Av*self.SOTE[4]*self.beta*self.g*self.h*self.phi**self.T*self.rho_S*self.yi[4]))/(4*self.C_star_20*self.O_Av*self.P_atm*self.phi**20) - (self.F[4]*self.SOTE[4]*self.phi**self.T*x47*self.yi[4]*ca.exp(-self.omega[4]*0.75*(x42+x43+x44+x45+x46)/1000))/(self.C_star_20*self.phi**20))) \
                        - (1-self.Y_H)/self.Y_H*(self.mu_H*x41/(self.K_S+x41)*x47/(x47+self.K_OH)*x44) - (4.57-self.Y_A)/self.Y_A*self.mu_A*x49/(self.K_NH+x49)*x47/(self.K_OA+x47)*x45,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x35-x48)  ) \
                        - (1-self.Y_H)/(2.86*self.Y_H)*(self.mu_H*x41/(self.K_S+x41)*self.K_OH/(x47+self.K_OH)*x48/(self.K_NO+x48)*self.eta_g*x44) + 1/self.Y_A*self.mu_A*x49/(self.K_NH+x49)*x47/(self.K_OA+x47)*x45,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x36-x49)  ) \
                        - self.i_XB*(self.mu_H*x41/(self.K_S+x41)*x47/(x47+self.K_OH)*x44) - self.i_XB*(self.mu_H*x41/(self.K_S+x41)*self.K_OH/(x47+self.K_OH)*x48/(self.K_NO+x48)*self.eta_g*x44) + (-self.i_XB-1/self.Y_A)*self.mu_A*x49/(self.K_NH+x49)*x47/(self.K_OA+x47)*x45 + self.k_a*x50*x44,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x37-x50)  ) \
                        - self.k_a*x50*x44 + x51/x43*self.k_h*(x43/x44)/(self.K_X+x43/x44)*(x47/(x47+self.K_OH)+self.eta_h*self.K_OH/(x47+self.K_OH)*x48/(x48+self.K_NO))*x44,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x38-x51)  ) \
                        + (self.i_XB-self.f_P*self.i_XP)*self.b_H*x44 + (self.i_XB-self.f_P*self.i_XP)*self.b_A*x45 - x51/x43*self.k_h*(x43/x44)/(self.K_X+x43/x44)*(x47/(x47+self.K_OH)+self.eta_h*self.K_OH/(x47+self.K_OH)*x48/(x48+self.K_NO))*x44,
                    1/self.Vol[4]*((Q0+Qa+Qr)*(x39-x52)  ) \
                        - self.i_XB/14*(self.mu_H*x41/(self.K_S+x41)*x47/(x47+self.K_OH)*x44) + ((1-self.Y_H)/(14*2.86*self.Y_H)-self.i_XB/14)*(self.mu_H*x41/(self.K_S+x41)*self.K_OH/(x47+self.K_OH)*x48/(self.K_NO+x48)*self.eta_g*x44) + (-self.i_XB/14-1/(7*self.Y_A))*self.mu_A*x49/(self.K_NH+x49)*x47/(self.K_OA+x47)*x45 + 1/14*self.k_a*x50*x44,

                    #Tank5
                    1/self.Vol[5]*((Q0+Qr)*x40-(Q0-self.Q_w)*x53  - (Qr+self.Q_w)*x53  ),
                    1/self.Vol[5]*((Q0+Qr)*x41-(Q0-self.Q_w)*x54 -(Qr+self.Q_w)*x54  ) \
                        - 1/self.Y_H*(self.mu_H*x54/(self.K_S+x54)*x60/(x60+self.K_OH)*x57) - 1/self.Y_H*(self.mu_H*x54/(self.K_S+x54)*self.K_OH/(x60+self.K_OH)*x61/(self.K_NO+x61)*self.eta_g*x57) + self.k_h*(x56/x57)/(self.K_X+x56/x57)*(x60/(x60+self.K_OH)+self.eta_h*self.K_OH/(x60+self.K_OH)*x61/(x61+self.K_NO))*x57,
                    1/self.Vol[5]*((Q0+Qr)*x42-(Q0-self.Q_w)*0-(Qr+self.Q_w)* x55  ),
                    1/self.Vol[5]*((Q0+Qr)*x43-(Q0-self.Q_w)*0-(Qr+self.Q_w)* x56  ) \
                        + (1-self.f_P)*self.b_H*x57 + (1-self.f_P)*self.b_A*x58 - self.k_h*(x56/x57)/(self.K_X+x56/x57)*(x60/(x60+self.K_OH) + self.eta_h*self.K_OH/(x60+self.K_OH)*x61/(x61+self.K_NO))*x57,
                    1/self.Vol[5]*((Q0+Qr)*x44-(Q0-self.Q_w)*0-(Qr+self.Q_w)* x57  ) \
                        + (self.mu_H*x54/(self.K_S+x54)*x60/(x60+self.K_OH)*x57) + (self.mu_H*x54/(self.K_S+x54)*self.K_OH/(x60+self.K_OH)*x61/(self.K_NO+x61)*self.eta_g*x57) - self.b_H*x57,
                    1/self.Vol[5]*((Q0+Qr)*x45-(Q0-self.Q_w)*0-(Qr+self.Q_w)* x58  ) \
                        + self.mu_A*x62/(self.K_NH+x62)*x60/(self.K_OA+x60)*x58 - self.b_A*x58,
                    1/self.Vol[5]*((Q0+Qr)*x46-(Q0-self.Q_w)*0-(Qr+self.Q_w)* x59  ) \
                        +  self.f_P*self.b_H*x57 + self.f_P*self.b_A*x58,
                    1/self.Vol[5]*((Q0+Qr)*x47-(Q0-self.Q_w)*x60 -(Qr+self.Q_w)*x60 \
                        + QA5*self.rho_A*self.O_Am*((ca.exp(-self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000))*(2*self.C_star_20*self.P_atm*self.phi**20*ca.exp(self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000)) - 2*self.phi**self.T*((1/self.phi**(2*self.T)*(4*self.C_star_20**2*self.P_atm**2*self.phi**40*ca.exp(2*self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000)) + self.C_star_T**2*self.F[5]**2*self.P_atm**2*self.SOTE[5]**2*self.beta**2*self.phi**(2*self.T)*self.yi[5]**2 + 4*self.F[5]**2*self.O_Av**2*self.P_atm**2*self.SOTE[5]**2*self.phi**(2*self.T)*x60**2*self.yi[5]**2 + 2*self.C_star_T**2*self.F[5]**2*self.O_Av*self.P_atm**2*self.SOTE[5]**2*self.beta**2*self.phi**(2*self.T)*self.yi[5]**2 + self.C_star_T**2*self.F[5]**2*self.O_Av**2*self.P_atm**2*self.SOTE[5]**2*self.beta**2*self.phi**(2*self.T)*self.yi[5]**2 - 4*self.C_star_T*self.F[5]**2*self.O_Av*self.P_atm**2*self.SOTE[5]**2*self.beta*self.phi**(2*self.T)*x60*self.yi[5]**2 - 4*self.C_star_T*self.F[5]**2*self.O_Av**2*self.P_atm**2*self.SOTE[5]**2*self.beta*self.phi**(2*self.T)*x60*self.yi[5]**2 + self.C_star_T**2*self.F[5]**2*self.O_Av**2*self.SOTE[5]**2*self.beta**2*self.g**2*self.h**2*self.phi**(2*self.T)*self.rho_S**2*self.yi[5]**2 + 4*self.C_star_20*self.C_star_T*self.F[5]*self.P_atm**2*self.SOTE[5]*self.beta*self.phi**self.T*self.phi**20*self.yi[5]*ca.exp(self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000)) + 8*self.C_star_20*self.F[5]*self.O_Av*self.P_atm**2*self.SOTE[5]*self.phi**self.T*self.phi**20*x60*self.yi[5]*ca.exp(self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000)) + 2*self.C_star_T**2*self.F[5]**2*self.O_Av*self.P_atm*self.SOTE[5]**2*self.beta**2*self.g*self.h*self.phi**(2*self.T)*self.rho_S*self.yi[5]**2 + 2*self.C_star_T**2*self.F[5]**2*self.O_Av**2*self.P_atm*self.SOTE[5]**2*self.beta**2*self.g*self.h*self.phi**(2*self.T)*self.rho_S*self.yi[5]**2 - 12*self.C_star_20*self.C_star_T*self.F[5]*self.O_Av*self.P_atm**2*self.SOTE[5]*self.beta*self.phi**self.T*self.phi**20*self.yi[5]*ca.exp(self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000)) - 4*self.C_star_T*self.F[5]**2*self.O_Av**2*self.P_atm*self.SOTE[5]**2*self.beta*self.g*self.h*self.phi**(2*self.T)*self.rho_S*x60*self.yi[5]**2 - 4*self.C_star_20*self.C_star_T*self.F[5]*self.O_Av*self.P_atm*self.SOTE[5]*self.beta*self.g*self.h*self.phi**self.T*self.phi**20*self.rho_S*self.yi[5]*ca.exp(self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000))))/4)**(1/2) + self.C_star_T*self.F[5]*self.P_atm*self.SOTE[5]*self.beta*self.phi**self.T*self.yi[5] + 2*self.F[5]*self.O_Av*self.P_atm*self.SOTE[5]*self.phi**self.T*x60*self.yi[5] + self.C_star_T*self.F[5]*self.O_Av*self.P_atm*self.SOTE[5]*self.beta*self.phi**self.T*self.yi[5] + self.C_star_T*self.F[5]*self.O_Av*self.SOTE[5]*self.beta*self.g*self.h*self.phi**self.T*self.rho_S*self.yi[5]))/(4*self.C_star_20*self.O_Av*self.P_atm*self.phi**20) - (self.F[5]*self.SOTE[5]*self.phi**self.T*x60*self.yi[5]*ca.exp(-self.omega[5]*(0.75*(x55+x56+x57+x58+x59)/1000)))/(self.C_star_20*self.phi**20))) \
                        - (1-self.Y_H)/self.Y_H*(self.mu_H*x54/(self.K_S+x54)*x60/(x60+self.K_OH)*x57) - (4.57-self.Y_A)/self.Y_A*self.mu_A*x62/(self.K_NH+x62)*x60/(self.K_OA+x60)*x58,
                    1/self.Vol[5]*((Q0+Qr)*x48-(Q0-self.Q_w)*x61 -(Qr+self.Q_w)*x61  ) \
                        + 1/self.Y_A*self.mu_A*x62/(self.K_NH+x62)*x60/(self.K_OA+x60)*x58 - (1-self.Y_H)/(2.86*self.Y_H)*(self.mu_H*x54/(self.K_S+x54)*self.K_OH/(x60+self.K_OH)*x61/(self.K_NO+x61)*self.eta_g*x57) ,
                    1/self.Vol[5]*((Q0+Qr)*x49-(Q0-self.Q_w)*x62 -(Qr+self.Q_w)*x62  ) \
                        - self.i_XB*(self.mu_H*x54/(self.K_S+x54)*x60/(x60+self.K_OH)*x57) - self.i_XB*(self.mu_H*x54/(self.K_S+x54)*self.K_OH/(x60+self.K_OH)*x61/(self.K_NO+x61)*self.eta_g*x57) + (-self.i_XB-1/self.Y_A)*self.mu_A*x62/(self.K_NH+x62)*x60/(self.K_OA+x60)*x58 + self.k_a*x63*x57,
                    1/self.Vol[5]*((Q0+Qr)*x50-(Q0-self.Q_w)*x63 -(Qr+self.Q_w)*x63  ) \
                        - self.k_a*x63*x57 + x64/x56*self.k_h*(x56/x57)/(self.K_X+x56/x57)*(x60/(x60+self.K_OH)+self.eta_h*self.K_OH/(x60+self.K_OH)*x61/(x61+self.K_NO))*x57,
                    1/self.Vol[5]*((Q0+Qr)*x51-(Q0-self.Q_w)*0-(Qr+self.Q_w)* x64  ) \
                        + (self.i_XB-self.f_P*self.i_XP)*self.b_H*x57 + (self.i_XB-self.f_P*self.i_XP)*self.b_A*x58 - x64/x56*self.k_h*(x56/x57)/(self.K_X+x56/x57)*(x60/(x60+self.K_OH)+self.eta_h*self.K_OH/(x60+self.K_OH)*x61/(x61+self.K_NO))*x57,
                    1/self.Vol[5]*((Q0+Qr)*x52-(Q0-self.Q_w)*x65 -(Qr+self.Q_w)*x65  ) \
                        - self.i_XB/14*(self.mu_H*x54/(self.K_S+x54)*x60/(x60+self.K_OH)*x57) + ((1-self.Y_H)/(14*2.86*self.Y_H)-self.i_XB/14)*(self.mu_H*x54/(self.K_S+x54)*self.K_OH/(x60+self.K_OH)*x61/(self.K_NO+x61)*self.eta_g*x57) + (-self.i_XB/14-1/(7*self.Y_A))*self.mu_A*x62/(self.K_NH+x62)*x60/(self.K_OA+x60)*x58 + 1/14*self.k_a*x63*x57
                
        ]
        
        # # Fouling mechanism
    
        # print(x55,x56,x57,x58,x59)                                                                                                # self.Rsc_)
        MLSS_ = 0.75*(x55+x56+x57+x58+x59)/1000
        J_ = (Q0-self.Q_w)/self.area #m3/m2.d    
        stickness_ = self.stickiness2 * (Ms_ * self.ksc >= 10e-6)
        stickness_ += (Ms_ * self.ksc < 10e-6) * (self.stickiness1 + self.ksc * Ms_ / 10e-6 * (self.stickiness2 - self.stickiness1))
        width = 6e-2  # channel width (mm->m)   
        Ax = 2 * width * self.area / self.yi[5]  # cross-sectional air sparging self.area (m2)  P. Buzatu 2017
        qa = QA5 / Ax
        mu_s_ = 1.05 * ca.exp(0.08 * MLSS_) *  self.mu_wPas * self.day_s
        # print('mlss', MLSS_)
        G0 = np.sqrt(qa * self.SDensity * self.g_m_day / mu_s_)
        # print('mu_s', mu_s_,'\t G', G0)
        G = G0
        Es_ = self.K1 * G / (24 * J_ + self.K1 * G)

        Vtf_ = J_ * ca.fmin(t, 0.9*self.cycle_time) /self.day_s
        dMsf = ca.if_else(
        t <= 0.9*self.cycle_time,
        J_ * MLSS_ * (1 - Es_) - (self.beta_f * (1 - stickness_) * G * x66 * x66 / (x66 + self.gamma1 * Vtf_ * (t/self.day_s))),
        -ca.sum1(self.beta_f * (1 - stickness_) * G * x66 * x66 / (x66 + 0.1 * self.gamma1 * Vtf_ * self.tf/self.day_s)))


        # dxdt2.append(dMsf)
        dxdt2 = ca.vertcat(*dxdt2, dMsf)
        self._ode_1s = ca.Function('ode_1s', [x, u, p, t], [dxdt2])
        return
    
    def _ini_ode_15(self):
        x = ca.MX.sym('x', self.N_Vars)
        u = ca.MX.sym('u', self.action_low.shape[0])
        p = ca.MX.sym('p', self.disturbance.shape[1])
        # t = ca.MX.sym('t', 1)
        x_t = x
        for t in range(1, self.cycle_time+1):
            # print(t)
            x_t = x_t + self._ode_1s(x_t, u, p, t) / self.day_s
        self.mbr_sim = ca.Function('mbr_sim', [x, u, p], [x_t], ['x', 'u', 'p'], ['x_next'])
        return

    def reset(self):
        # Reset state to initial values
        Init_Vals1 = np.array([30.00, 2.25, 2678.62, 82.52, 2699.15, \
                    233.30, 1781.17, 0.01, 4.09, 8.57, \
                    1.08, 5.38, 5.07])
        Init_Vals2 = np.array([30.00, 1.31, 2678.62, 76.19, 2697.86, \
                            233.07, 1782.50, 0.01, 1.48, 9.22, \
                            0.68, 5.16, 5.30])
        Init_Vals3 = np.array([30.00, 0.85, 3554.43, 65.13, 3573.19, \
                            311.13, 2372.10, 2.46, 10.08, 1.58, \
                            0.65, 4.73, 4.14])
        Init_Vals4 = np.array([30.00, 0.77, 3554.43, 59.35, 3572.44, \
                            311.33, 2372.11, 2.19, 11.54, 0.33, \
                            0.63, 4.40, 3.95])
        Init_Vals5 = np.array([30.00, 0.67, 4722.18, 67.25, 4739.59, \
                            413.41, 3155.87, 8.11, 12.57, 0.07, \
                            0.58, 5.14, 3.85])
                

        Init_Vals = np.concatenate((Init_Vals1, Init_Vals2, Init_Vals3, Init_Vals4, Init_Vals5, np.zeros(1)))
        self.state = Init_Vals
        self.count = 0
        self.current_step = 0
        self.Rp = 10e11
        
        self.disturbance = np.loadtxt('envs/Inf_dry_constQ.txt') 
        self.disturbance = np.hstack((self.disturbance[:, -1].reshape(-1, 1), self.disturbance[:, 1:-1]))
        self.disturbance = np.hstack((self.disturbance, np.zeros((self.disturbance.shape[0], 1))))
        self.p = self.disturbance
        return self.state

    def get_action(self):  # for KLa5 and Qa-recycle flow rate

        # if self.count % self.action_sample_period == 0:
        #     if self.count == 0:
        #         self.a_holder = self.action_space.sample()
        #     else:
        #         a_new = self.action_space.sample()
        #         self.a_holder = np.clip(a_new, self.a_holder - self.action_high*0.2, self.a_holder + self.action_high*0.2)
        # a = self.a_holder + np.random.normal(np.zeros_like(self.action_high),
        #                                      self.action_high * 0.01)  # TODO: PAY ATTENTION
        # a = np.clip(a, self.action_low, self.action_high)
        
        # # a = np.array([21450, 32180.0 * 3])
        # self.count += 1

        if self.current_step % self.action_sample_period==0:
            if self.count == 0:
                self.a_holder = self.action_space.sample()
            else:
                self.target_a = self.action_space.sample()  # New target action
                self.transition_steps = np.random.randint(15, 20)  # 5 to 10 steps
                self.step_size = np.clip(
                    (self.target_a - self.a_holder) / self.transition_steps,
                    -self.action_high * 0.15,
                    self.action_high * 0.15
                )

                self.current_step = 0  # Reset step counter
                self.action_sample_period = np.random.randint(40, 60)

        # Smooth transition in the first `transition_steps` of the period
        if self.count > self.action_sample_period:
            if self.current_step < self.transition_steps:
                self.a_holder += self.step_size
        self.current_step += 1  # Move to the next step

        a = self.a_holder + np.random.normal(np.zeros_like(self.action_high), self.action_high * 0.01)
        a = np.clip(a, self.action_low, self.action_high)

        self.count += 1



        return a

    def _ini_econ_cost(self):
        x = ca.MX.sym('x', self.N_Vars)
        u = ca.MX.sym('u', self.action_low.shape[0])
        p = ca.MX.sym('p', self.disturbance.shape[1])
        Rp = ca.MX.sym('Rp', 1)
        Q0 = p[0]  # Jan 26, question, check if this is the 1st row of 1st column
        Qe = Q0 - self.Q_w
        Q_r = u[-1]
        Q_int = u[-1]


        # define parameter value
 
        self.Vol[1] = 1500
        self.Vol[2] = 1500
        self.Vol[3] = 1500
        self.Vol[4] = 1500
        self.Vol[5] = 1500

        KLa3 = u[0] * 24
        KLa4 = u[1] * 24
        KLa5 =  u[2] * 24

        SNHe = x[61]
        SNDe = x[62]
        SSe = x[53]
        SIe = x[52]
        SNOe = x[60]

        # XNDe XBHe XBAe XPe XIe XSe
        XI5 = x[54]
        XS5 = x[55]
        XBH5 = x[56]
        XBA5 = x[57]
        XP5 = x[58]
        XND5 = x[63]
        
        # Xe = x[74]
        Xf = 0.75 * (XI5 + XS5 + XBH5 + XBA5 + XP5)
        Xe = 0
        XNDe = (Xe / Xf) * XND5
        XBHe = (Xe / Xf) * XBH5
        XBAe = (Xe / Xf) * XBA5
        XPe = (Xe / Xf) * XP5
        XIe = (Xe / Xf) * XI5
        XSe = (Xe / Xf) * XS5

        ## EQ ---------------------------------------------------------------##
        SNKje = SNHe + SNDe + XNDe + self.i_XB * (XBHe + XBAe) + self.i_XP * (XPe + XIe)
        CODe = SSe + SIe + XSe + XIe + XBHe + XBAe + XPe
        BOD5e = 0.25 * (SSe + XSe + (1 - self.f_P) * (XBHe + XBAe))
        TSSe = Xe

        # KLa5 = 94.5898 * u[0]
        # Qa = 32834.8542 * u[1]

        # KLa5 = u[0]
        # Qa = u[1]
        # print(TSSe, CODe, SNKje, SNOe, BOD5e)
        EQk = (2 * TSSe + CODe + 30 * SNKje + 10 * (SNOe) + 2 * BOD5e) * Qe / 1000

        ## OCI --------------------------------------------------------------##
        TSSak = 0.75 * (self.Vol[1] * (x[2] + x[3] + x[4] + x[5] + x[6]) + \
                        self.Vol[2] * (x[15] + x[16] + x[17] + x[18] + x[19]) + \
                        self.Vol[3] * (x[28] + x[29] + x[30] + x[31] + x[32]) + \
                        self.Vol[4] * (x[41] + x[42] + x[43] + x[44] + x[45]) + \
                        self.Vol[5] * (x[54] + x[55] + x[56] + x[57] + x[58]))

        # TSSsk is for clarifier, which is not in BSM-MBR
        # TSSsk = zm * A * (x[65] + x[66] + x[67] + x[68] + x[69] + x[70] + x[71] + x[72] + x[73] + x[74])
        # TSSk = TSSak + TSSsk
        TSSk = TSSak
        # waste stream
        # weighted_ratio = (Q0 +  Q_r) / ( Q_r +  self.Q_w)
        # SPk = (TSSk - TSS0 + Xf * weighted_ratio *  self.Q_w) / 1000
        SPk = Xf *  self.Q_w / 1000  # kg/d #Instantaneous sludge production (kg/d)
        # PEk = 0.004 * Qa + 0.008 * Qr + 0.05 * Qw
        # print(Rp)
        tmp =  self.mu_wPas*(1e15*p[-1]+ self.Rm + Rp)*Qe/ self.area /self.day_s # Pa
        # tmp =  self.mu_wPas*(self.Rm + Rp)*Qe/ self.area /self.day_s
        R_total = 1e15*p[-1]+ self.Rm + Rp
        # print('tmp', tmp)
        PEk = 0.0075 * ( Q_r +  Q_int) + 0.05 *  self.Q_w + Qe * tmp / 3.6e6 / 0.70 #Mannina2013
        # PEk = 0.016 * ( Q_r +  Q_int+  self.Q_w) + Qe * tmp / 3.6e6 / 0.70 #Mannina2013
        # PEk = 0.0075 * ( Q_r+ Q_int) + 0.05 *  self.Q_w + Qe*0.075 ## no fouling

        # AEk = SO_star / 1.8 / 1000 * (self.Vol[3] * self.QA3 + self.Vol[4] * self.QA4 + self.Vol[5] * KLa5)
        AEk = ( KLa3 +  KLa4) * 0.025 +  KLa5 * 0.019
        MEk = 24 * 0.008 * (self.Vol[1] + self.Vol[2])  ###Coefficients from Maere2011

        OCIk = AEk + PEk + 5 * SPk + MEk  #[15594.565] [3083.6077]
        cost = EQk + self.wOCI * OCIk
        # s_attention = [self.state[21], self.state[59]]
        # cost = ca.norm_2(s_attention - self.xs)
        # components = dict(EQk=EQk, OCIk=OCIk, TSSe=TSSe, CODe=CODe, SNKje=SNKje, SNOe=SNOe, BOD5e=BOD5e, AEk=AEk, PEk=PEk, SPk=SPk, MEk=MEk)
        # components = {'EQ':[TSSe, CODe, SNKje, SNOe, BOD5e], 'OCI':[AEk, PEk, SPk, MEk], 'R':[1e15*p[-1], self.Rm, self.Rp, R_total]}
        self._econ_cost = ca.Function('econ_cost', [x, u, p, Rp], [cost, EQk, OCIk, tmp, TSSe, CODe, SNKje, SNOe, BOD5e,AEk, PEk, SPk, MEk])
        return #EQk + 0.3 * OCIk #, EQk, OCIk, (TSSe, CODe, SNKje, SNOe, BOD5e), (AEk, PEk, SPk, MEk), (1e15*p[-1], self.Rm, self.Rp, R_total)   #+ 0.01 * mpc.mtimes(Du.self.T, Du)
    
    def step(self, u, i):
        r = self.mbr_sim(x=self.state, u=u, p=self.p[i,:])
        self.state = r['x_next'].full().flatten()
        
        self.Rp += self.rp*(self.p[i, 0] - self.Q_w) / self.area * self.tf_d
        self.p[:,-1] += self.state[-1]
        cost, EQ, OCI, tmp, TSSe, CODe, SNKje, SNOe, BOD5e,AEk, PEk, SPk, MEk = self._econ_cost(self.state, u, self.p[i,:], self.Rp)
        components = {'EQ':[TSSe, CODe, SNKje, SNOe, BOD5e], 'OCI':[AEk, PEk, SPk, MEk]}
        #tracking cost

        s_attention = [self.state[59],self.state[21]] 
        # ref_attention = []
        cost = np.linalg.norm(s_attention - self.xs)
        
        if i == self.p.shape[0] - 1:
            done = True
        else:
            done = False
        info = {
            'EQ': EQ,
            'OCI': OCI,
            'tmp': tmp,
            'components': components
        }

        self.x = self.state
        return self.state, cost, done , info
    
    def _ini_step(self):
        x = ca.MX.sym('x', self.N_Vars)
        u = ca.MX.sym('u', self.action_low.shape[0])
        p = ca.MX.sym('p', self.disturbance.shape[1])
        # i = ca.MX.sym('i', 1)
        Rp = ca.MX.sym('Rp', 1)

        # next state
        r = self.mbr_sim(x=x, u=u, p=p)
        x_next = r['x_next']
        
        # update parameters for cost function
        add_Rp = self.rp*(p[0] - self.Q_w) / self.area * self.tf_d
        Rp_ = Rp + add_Rp # pore fouling
        Ms_ = p[-1] + x_next[-1] # cake fouling
        p_updated = ca.vertcat(p[:-1], Ms_)

        # cost
        cost, EQk, OCIk, tmp, TSSe, CODe, SNKje, SNOe, BOD5e,AEk, PEk, SPk, MEk = self._econ_cost(x_next, u, p_updated, Rp_)

        self.step_sym = ca.Function('step', [x, u, p, Rp], [x_next, cost, Rp_])
        return

    

def main():
    env = MBRGymEnv(wOCI=0.05)

    circle = 50 # int(96 * env.t_operate)
    inputs = np.loadtxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/action-MK-60kPa-Nc=10-Np=30.txt')
    circle = len(inputs)
    KLa4 = [0, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000, 5500, 6000, 6500, 7000]
    KLa5 = np.arange(5000, 25000, 2000)
    # circle = inputs.shape[0]
    time_cir = 1 # len(KLa5)
    # save_all   = np.zeros([time_cir,circle,6])
    save_state = np.zeros([circle * time_cir+1, env.state.shape[0]])
    save_input = np.zeros([circle * time_cir, env.action_high.shape[0]])
    all_cost = np.zeros([circle * time_cir, 3])
    EQI = np.zeros([circle * time_cir, 5])
    OCI = np.zeros([circle * time_cir, 4])
    tmps = [3326]
    # R = np.zeros([circle * time_cir, 4])
    
    for t_c in range(time_cir):
        np.random.seed(t_c)
        save_state[t_c * (circle), :] = env.reset()
        print("reset!")

        for i in tqdm(range(circle)):
            # action = env.get_action()
            # action = inputs[i, :4]
            if i < circle//2:
                action = [4500, 2500, 20000, 32180.0*5]
            else:
                action = [4500, 2500, 38012, 32180.0*5]
            save_input[t_c * (circle) + i, :] = action
            x, cost, done,EQIk, OCIk,tmp, components = env.step(action, i)
            tmps.append(tmp.full().flatten()[0])
            if i != circle - 1:
                save_state[t_c * (circle) + i + 1, :] = env.state
                # print(env.P_ / 1000.0)
                all_cost[t_c * (circle) + i + 1, :] = np.array([cost,EQIk, OCIk]).flatten()
                EQI[t_c * (circle) + i + 1, :] = np.array(components['EQ']).flatten()
                OCI[t_c * (circle) + i + 1, :] = np.array(components['OCI']).flatten()
                # R[t_c * (circle) + i + 1, :] = components['R']
            if done == True:
                break
    # save_state[:,-1] = np.array(tmps).flatten()
    # used_disturbances = np.array([14,2, 3, 4, 5, 10, 11, 12])
    # pred_index = set(np.arange(66))
    # cont_index = {0, 13, 26, 39, 52}
    # pred_index = np.array(list(pred_index - cont_index))
    # save_state = np.hstack((save_state[:,pred_index], env.p[:,used_disturbances]))
    
#     # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/result_fixMsf.txt', save_all)
#     # scipy.io.savemat('tmp.mat', {'fouling':save_all})
    # np.save('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/states-MK-partial.npy', save_state)
    # np.save('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/inputs-MK-partial.npy', save_input)
    # np.save('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/costs-MK-partial.npy', all_cost)

    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/states-MK-partial.csv', save_state)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/inputs-MK-partial.csv', save_input)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/costs-MK-partial.csv', all_cost)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/EQI-MK-partial.csv', EQI)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/OCI-MK-partial.csv', OCI)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/TMP-MK-partial.csv', np.array(tmps))
#     # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/R-KLa5=21450.csv', R)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(np.diff(np.array(tmps)))
    plt.ylabel('dTMP (Pa)')
    plt.grid()

    plt.figure()
    plt.plot(all_cost[1:, 0])
    plt.ylabel('Cost')
    plt.grid()

    plt.figure()
    plt.plot(all_cost[1:, 1])
    plt.ylabel('EQI')
    plt.grid()

    plt.figure()
    plt.plot(all_cost[1:, 2])
    plt.ylabel('OCI')
    plt.grid()

    plt.figure()
    plt.plot(save_state[:-1, 61])
    plt.ylabel('$S_{NH}$')
    plt.grid()

    plt.figure()
    plt.plot(np.array(tmps))
    plt.ylabel('$TMP$')
    plt.grid()
    # plt.figure()
    # plt.plot(np.sum(save_state[:-1, 60:63], axis=1))
    # plt.ylabel('$TN$')
    # plt.grid()

    plt.show()
# def main():
#     env = MBRGymEnv(wOCI=0.05)

#     circle = int(96 * env.t_operate)
#     KLa5 = [20000]
#     time_cir = len(KLa5)
#     # save_all   = np.zeros([time_cir,circle,6])
#     save_state = np.zeros([circle * time_cir+1, env.state.shape[0]])
#     save_input = np.zeros([circle * time_cir, env.action_high.shape[0]])
#     all_cost = np.zeros([circle * time_cir, 3])
#     EQI = np.zeros([circle * time_cir, 5])
#     OCI = np.zeros([circle * time_cir, 4])
    
#     # R = np.zeros([circle * time_cir, 4])
#     import matplotlib.pyplot as plt
#     f1, ax1 = plt.subplots()
#     f2, ax2 = plt.subplots()
#     for t_c in range(time_cir):
#         tmps = []
#         np.random.seed(t_c)
#         save_state[t_c * (circle), :] = env.reset()
#         print("reset!")

#         for i in tqdm(range(circle)):
#             action = env.get_action()
#             # action = [4250, 2250, KLa5[t_c], 96540]
#             save_input[t_c * (circle) + i, :] = action
#             x, cost, done ,EQk, OCIk, tmp, components = env.step(action, i)
#             tmps.append(tmp.full().flatten())
#             if i != circle - 1:
#                 save_state[t_c * (circle) + i + 1, :] = env.state
#                 # print(env.P_ / 1000.0)
#                 all_cost[t_c * (circle) + i + 1, :] = np.array([cost,EQk, OCIk]).flatten()
#                 EQI[t_c * (circle) + i + 1, :] = np.array(components['EQ']).flatten()
#                 OCI[t_c * (circle) + i + 1, :] = np.array(components['OCI']).flatten()
#                 # R[t_c * (circle) + i + 1, :] = components['R']
#             if done == True:
#                 break
#         ax1.plot(np.array(tmps)/1000, label='KLa5 = %d' % KLa5[t_c])
#         ax2.plot(np.array(tmps)[1:]/1000 - np.array(tmps)[:-1]/1000, label='KLa5 = %d' % KLa5[t_c])
#         ax1.legend()
#         ax2.legend()
#     ax1.set_ylabel('TMP (kPa)')
#     ax2.set_ylabel('dTMP (kPa)')
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/result_fixMsf.txt', save_all)
    # scipy.io.savemat('tmp.mat', {'fouling':save_all})
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/states_test.csv', save_state)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/inputs_test.csv', save_input)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/costs_test.csv', all_cost)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/EQI_test.csv', EQI)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/OCI_test.csv', OCI)
    # np.savetxt('C:/Users/thumi/OneDrive - Nanyang Technological University/codes/meta learning w membrane fouling/data/test/R-KLa5=21450.csv', R)

    
    
    # plt.grid()

    # plt.figure()
    # plt.plot(all_cost[1:, 0])
    # plt.ylabel('Cost')
    # plt.grid()

    # plt.figure()
    # plt.plot(all_cost[1:, 1])
    # plt.ylabel('EQI')
    # plt.grid()

    # plt.figure()
    # plt.plot(all_cost[1:, 2])
    # plt.ylabel('OCI')
    # plt.grid()

    # plt.figure()
    # plt.plot(save_state[:-1, 61])
    # plt.ylabel('$S_{NH}$')
    # plt.grid()

    # plt.figure()
    # plt.plot(np.sum(save_state[:-1, 60:63], axis=1))
    # plt.ylabel('$TN$')
    # plt.grid()

    # plt.show()

if __name__ == '__main__':
    main()
     