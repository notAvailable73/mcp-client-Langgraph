from mcp.server.fastmcp import FastMCP  
mcp =FastMCP("dateTime") 
from datetime import datetime
@mcp.tool
def get_present_date() -> str:
    """Get the current date and time.

    Returns:
        str: Current date and time in ISO format
    """
    return datetime.now().isoformat()


if __name__ =="__main__":
    mcp.run(transport="stdio")