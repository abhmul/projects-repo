import argparse
import asyncio
from pyiqvia import Client


parser = argparse.ArgumentParser()
parser.add_argument('zipcode', help="current zipcode for data")


async def historic(client: Client) -> tuple:
    allergies = await client.allergens.historic
    asthma = await client.asthma.historic
    disease = await client.disease.historic
    return (allergies, asthma, disease)


if __name__ == "__main__":
    args = parser.parse_args()
    client = Client(args.zipcode)
    allergies, asthma, disease = historic(client)
    