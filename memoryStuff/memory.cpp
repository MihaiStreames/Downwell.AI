#include <Windows.h>
#include <TlHelp32.h>
#include <iostream>
#include <vector>

DWORD getProcId(const char* procName)
{
    DWORD procId = 0;
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPPROCESS, 0);

    if (hSnap != INVALID_HANDLE_VALUE)
    {
        PROCESSENTRY32 procEntry;
        procEntry.dwSize = sizeof(procEntry);

        if (Process32First(hSnap, &procEntry))
        {
            do
            {
                if (!_stricmp(procEntry.szExeFile, procName))
                {
                    procId = procEntry.th32ProcessID;
                    break;
                }
            } while (Process32Next(hSnap, &procEntry));
        }
    }
    CloseHandle(hSnap);
    return procId;
}

uintptr_t getModuleBaseAddress(DWORD procId, const char* modName)
{
    uintptr_t modBaseAddr = 0;
    HANDLE hSnap = CreateToolhelp32Snapshot(TH32CS_SNAPMODULE | TH32CS_SNAPMODULE32, procId);

    if (hSnap != INVALID_HANDLE_VALUE)
    {
        MODULEENTRY32 modEntry;
        modEntry.dwSize = sizeof(modEntry);

        if (Module32First(hSnap, &modEntry))
        {
            do
            {
                if (!_stricmp(modEntry.szModule, modName))
                {
                    modBaseAddr = (uintptr_t)modEntry.modBaseAddr;
                    break;
                }
            } while (Module32Next(hSnap, &modEntry));
        }
    }
    CloseHandle(hSnap);
    return modBaseAddr;
}

uintptr_t findDMAAddy(HANDLE hProc, uintptr_t ptr, std::vector<unsigned int> offsets)
{
    uintptr_t addr = ptr;
    for (unsigned int i = 0; i < offsets.size(); ++i)
    {
        ReadProcessMemory(hProc, (BYTE*)addr, &addr, sizeof(addr), 0);
        addr += offsets[i];
    }
    return addr;
}

int main()
{
    const char* procName = "downwell.exe";
    DWORD procId = 0;

    while (!procId)
    {
        procId = getProcId(procName);
        Sleep(30);
    }

    uintptr_t moduleBase = getModuleBaseAddress(procId, procName);

    HANDLE hProcess = OpenProcess(PROCESS_ALL_ACCESS, 0, procId);

    // The offsets will depend on the game memory structure
    std::vector<unsigned int> hpOffsets = {0x708, 0xC, 0x24, 0x10, 0x9C0, 0x390};
    uintptr_t hpBase = moduleBase + 0x004A5E50;

    std::vector<unsigned int> ammoOffsets = {0x10, 0x1F0, 0x78, 0x24, 0x10, 0xF24, 0x3C0};
    uintptr_t ammoBase = moduleBase + 0x00536180;

    std::vector<unsigned int> gemOffsets = {0x24, 0x10, 0x330, 0xE0, 0x50, 0x9A8, 0x350};
    uintptr_t gemBase = moduleBase + 0x00757BF0;

    while (true)
    {
        uintptr_t hpAddr = findDMAAddy(hProcess, hpBase, hpOffsets);
        uintptr_t ammoAddr = findDMAAddy(hProcess, ammoBase, ammoOffsets);
        uintptr_t gemAddr = findDMAAddy(hProcess, gemBase, gemOffsets);
        int hpValue;
        int ammoValue;
        int gemValue;
        ReadProcessMemory(hProcess, (BYTE*)hpAddr, &hpValue, sizeof(hpValue), nullptr);
        ReadProcessMemory(hProcess, (BYTE*)ammoAddr, &ammoValue, sizeof(ammoValue), nullptr);
        ReadProcessMemory(hProcess, (BYTE*)gemAddr, &gemValue, sizeof(gemValue), nullptr);

        std::cout << "Current HP: " << hpValue << std::endl;
        std::cout << "Current Ammo: " << ammoValue << std::endl;
        std::cout << "Current Gems: " << gemValue << std::endl;

        Sleep(1000);
    }

    return 0;
}