from argparse import ArgumentParser
import pickle

import gym
import minerl

from openai_vpt.agent import MineRLAgent, ENV_KWARGS

def main(model, weights, env):
    env = gym.make(env)
    print("---Loading model---")
    agent_parameters = pickle.load(open(model, "rb"))
    policy_kwargs = agent_parameters["model"]["args"]["net"]["args"]
    pi_head_kwargs = agent_parameters["model"]["args"]["pi_head_opts"]
    pi_head_kwargs["temperature"] = float(pi_head_kwargs["temperature"])
    agent = MineRLAgent(env, policy_kwargs=policy_kwargs, pi_head_kwargs=pi_head_kwargs)
    agent.load_weights(weights)

    print("---Launching MineRL enviroment (be patient)---")
    obs = env.reset()
    step=0
    while True:
        step+=1
        minerl_action = agent.get_action(obs)
        # ESC is not part of the predictions model.
        # For baselines, we just set it to zero.
        # We leave proper execution as an exercise for the participants :)
        minerl_action["ESC"] = 0
        obs, reward, done, info = env.step(minerl_action)


        # if step>1000:
        #     reward=-reward

        env.render()


if __name__ == "__main__":
    parser = ArgumentParser("Run pretrained models on MineRL environment")

    parser.add_argument("--weights", type=str, default="train/ori/MineRLBasaltFindCavegail26.weights", help="Path to the '.weights' file to be loaded.")
    parser.add_argument("--model", type=str, default="train/foundation-model-1x.model", help="Path to the '.model' file to be loaded.")
    parser.add_argument("--env", type=str, default="MineRLBasaltFindCave-v0")

    args = parser.parse_args()

    main(args.model, args.weights, args.env)
