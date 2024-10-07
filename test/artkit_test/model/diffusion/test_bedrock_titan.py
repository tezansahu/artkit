import os
from collections.abc import Generator
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import boto3
import pytest
from aiohttp import ClientResponseError
from moto import mock_aws

from artkit.model.diffusion.bedrock import TitanBedrockDiffusionModel
from artkit.model.util import RateLimitException

#######################################################################################
#                                     Constants                                       #
#######################################################################################
MODEL_ID = "amazon.titan-image-generator-v1"
REGION = "us-east-1"

MESSAGE = "A cat with wings"


@pytest.mark.asyncio
async def test_text_to_image(
    mock_bedrock_diffusion_chat: TitanBedrockDiffusionModel,
) -> None:
    with patch(
        "artkit.model.diffusion.bedrock.base._base.ClientSession.__aenter__"
    ) as MockClientSession:
        mock_post = Mock()
        mock_post.json = AsyncMock(return_value={"images": ["dGVzdA=="]})
        mock_post.text = AsyncMock()
        mock_post.return_value.status_code = 200

        mock_connection = AsyncMock()
        mock_connection.post.return_value = mock_post
        MockClientSession.return_value = mock_connection

        response = await mock_bedrock_diffusion_chat.text_to_image(text=MESSAGE)
        assert response[0].data.decode("ascii") == "test"


@pytest.mark.asyncio
async def test_rate_limit_error(
    mock_bedrock_diffusion_chat: TitanBedrockDiffusionModel,
) -> None:
    with patch(
        "artkit.model.diffusion.bedrock.base._base.ClientSession.__aenter__"
    ) as MockClientSession:
        # Set up the mock connection object
        mock_connection = AsyncMock()

        def raise_rate_limit_error() -> None:
            err = ClientResponseError(
                request_info=AsyncMock(),
                history=AsyncMock(),
                status=429,
                message="Rate limit exceeded",
            )
            raise err

        mock_connection.post.return_value.raise_for_status = raise_rate_limit_error
        MockClientSession.return_value = mock_connection

        with pytest.raises(RateLimitException):
            await mock_bedrock_diffusion_chat.text_to_image(text=MESSAGE)


#######################################################################################
#                                     FIXTURES                                        #
#######################################################################################
@pytest.fixture(scope="function")
def mock_bedrock_diffusion_chat(
    aws_credentials: Any,
) -> Generator[TitanBedrockDiffusionModel, None, None]:
    with mock_aws():
        # Mock the AWS credentials
        boto3.client("sts").get_caller_identity()

        yield TitanBedrockDiffusionModel(
            model_id=MODEL_ID,
            region=REGION,
            max_retries=2,
            initial_delay=0.1,
            exponential_base=1.5,
        )


@pytest.fixture(scope="function")
def aws_credentials() -> None:
    """Mocked AWS Credentials for moto."""
    os.environ["AWS_ACCESS_KEY_ID"] = "testing"
    os.environ["AWS_SECRET_ACCESS_KEY"] = "testing"
    os.environ["AWS_SECURITY_TOKEN"] = "testing"
    os.environ["AWS_SESSION_TOKEN"] = "testing"
    os.environ["AWS_DEFAULT_REGION"] = "us-east-1"
